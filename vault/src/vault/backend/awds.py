import importlib
import os
import re
import sys
import tarfile
import traceback
from collections.abc import Callable, Iterable, Iterator, Sequence
from functools import partial
from typing import Any, Optional, Union

import megfile
import webdataset as wds
from webdataset.autodecode import IMAGE_EXTENSIONS
from webdataset.filters import pipelinefilter

IMAGE_EXTENSIONS_LISTSTR = ";".join([*IMAGE_EXTENSIONS, "avif"])


def distributed_identifier() -> str:
    rank, world_size, worker, num_workers = wds.utils.pytorch_worker_info()
    return f"#distributed(rank={rank}, world_size={world_size}, worker={worker}, num_workers={num_workers})"


def print_exception_and_continue(exn: Exception):
    print(distributed_identifier(), file=sys.stderr)
    if hasattr(exn, "note"):
        print(exn.note, file=sys.stderr)  # type: ignore
    traceback.print_exception(exn, file=sys.stderr)

    return True


def _add_note_to_exception(exn: Exception, note: str):
    exn.note = note  # type: ignore
    return exn


def register_gopen_bachend(protocol, function):
    importlib.import_module("webdataset.gopen").gopen_schemes[protocol] = function


def gopen_s3(url, mode="rb", bufsize=8192, **kwargs):
    import fsspec

    return fsspec.filesystem("s3", **kwargs).open(url, mode=mode)


def gopen_s3_pipe(url, mode="rb", bufsize=8192, endpoint_url=None):
    url = "s3://" + url.split("://")[1]
    if endpoint_url is not None:
        url = f"pipe: aws --endpoint-url={endpoint_url} s3 cp {url} -"
    else:
        url = f"pipe: aws s3 cp {url} -"
    from webdataset.gopen import gopen_pipe

    return gopen_pipe(url, mode=mode, bufsize=bufsize)


def gopen_megfile(url, mode="rb", bufsize=8192, **kwargs):
    return megfile.smart_open(url, mode=mode)  # type: ignore


def _reset_default_gopen_bachend_as_megfile():
    gopen = importlib.import_module("webdataset.gopen")

    gopen.gopen_schemes["__default__"] = gopen_megfile


_reset_default_gopen_bachend_as_megfile()

# Open URLs and yield a stream of url+stream pairs.
url_opener = pipelinefilter(wds.tariterators.url_opener)


def tar_file_iterator(
    fileobj: tarfile.TarFile,
    skip_meta: Optional[str] = r"__[^/]*__($|/)",
    handler: Callable[[Exception], bool] = wds.reraise_exception,
    select_files: Optional[Callable[[str], bool]] = None,
    rename_files: Optional[Callable[[str], str]] = None,
    meta_prefix="__",
    meta_suffix="__",
) -> Iterator[dict[str, Any]]:
    """Iterate over tar file, yielding filename, content pairs for the given tar stream.

    Args:
        fileobj: the tar file stream.
        skip_meta: regexp for keys that are skipped entirely. Defaults to r"__[^/]*__($|/)".
        handler: exception handler. Defaults to reraise_exception.
        select: predicate for selecting files. Defaults to None.

    Yields:
        a stream of samples.
    """

    stream = tarfile.open(fileobj=fileobj, mode="r|*")  # type: ignore  # noqa: SIM115
    for tarinfo in stream:
        fname = tarinfo.name
        try:
            if not tarinfo.isreg() or fname is None:
                continue
            if (
                "/" not in fname
                and fname.startswith(meta_prefix)
                and fname.endswith(meta_suffix)
            ):
                # skipping metadata for now
                continue
            if skip_meta is not None and re.match(skip_meta, fname):
                continue
            if rename_files:
                fname = rename_files(fname)
            if select_files is not None and not select_files(fname):
                continue
            data = stream.extractfile(tarinfo).read()
            result = dict(
                fname=fname, data=data, size=tarinfo.size, offset=tarinfo.offset_data
            )
            yield result
            stream.members = []
        except Exception as exn:
            if hasattr(exn, "args") and len(exn.args) > 0:
                exn.args = (str(exn.args[0]) + " @ " + str(fileobj),) + exn.args[1:]
            if handler(exn):
                continue
            else:
                break
    del stream


def _expand_tarfile(
    source,
    select_files: Optional[Callable[[str], bool]] = None,
    rename_files: Optional[Callable[[str], str]] = None,
    eof_value: Optional[Any] = None,
    handler: Callable[[Exception], bool] = wds.reraise_exception,
    tar_index_handler: Optional[Callable] = None,
):
    assert isinstance(source, dict)
    assert "stream" in source

    url = source["url"]
    local_path = source.get("local_path")

    tar_index = None
    if tar_index_handler is not None:
        tar_index = {"url": url, "files": {}}

    for sample in tar_file_iterator(
        source["stream"],
        handler=handler,
        select_files=select_files,
        rename_files=rename_files,
    ):
        assert isinstance(sample, dict) and "data" in sample and "fname" in sample
        sample["__url__"] = url
        if local_path is not None:
            sample["__local_path__"] = local_path
        yield sample

        if tar_index_handler is not None and tar_index is not None:
            tar_index["files"][sample["fname"]] = dict(
                offset=sample["offset"], size=sample["size"]
            )
    # we yield an EOF marker at the end of each shard so that
    # samples from different shards don't get mixed up
    if eof_value is not None:
        yield eof_value

    if tar_index_handler is not None:
        tar_index_handler(tar_index)


def _split_path(p):
    prefix, suffix = os.path.splitext(p)
    return prefix, suffix[1:]


def valid_sample(sample: dict[str, Any] | None) -> bool:
    """Check whether a sample is valid.

    Args:
        sample: a

    Returns:
        boolean indicating whether the sample is valid.
    """
    return (
        sample is not None
        and isinstance(sample, dict)
        and len(list(sample.keys())) > 0
        and not sample.get("__bad__", False)
    )


def group_by_keys(  # noqa: C901
    data: Iterable[dict[str, Any]],
    keys: Callable[[str], tuple[str, str]] = _split_path,
    lcase: bool = True,
    suffixes: Optional[set[str]] = None,
    handler: Callable[[Exception], bool] = wds.reraise_exception,
) -> Iterator[dict[str, Any]]:
    """Group tarfile contents by keys and yield samples.

    Args:
        data: iterator over tarfile contents
        keys: function that takes a file name and returns a key and a suffix.
        lcase: whether to lowercase the suffix.
        suffixes: list of suffixes to keep.
        handler: exception handler.

    Raises:
        ValueError: raised if there are duplicate file names in the tar file.

    Yields:
        iterator over samples.
    """
    current_sample = None
    for filesample in data:
        try:
            assert isinstance(filesample, dict)
            if filesample == {}:
                if current_sample is not None and valid_sample(current_sample):
                    yield current_sample
                current_sample = None
                continue
            fname, value = filesample["fname"], filesample["data"]
            prefix, suffix = keys(fname)

            if prefix is None:
                continue
            if lcase:
                suffix = suffix.lower()
            if current_sample is None or prefix != current_sample["__key__"]:
                if current_sample is not None and valid_sample(current_sample):
                    yield current_sample
                current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
            if suffix in current_sample:
                raise ValueError(
                    f"{fname}: duplicate file name in tar file {current_sample.get('__url__')}"
                    + f"`{suffix}` {current_sample.keys()}"
                )
            if suffixes is None or suffix in suffixes:
                current_sample[suffix] = value
            local_path = filesample.get("__local_path__")
            if local_path is not None:
                current_sample["__local_path__"] = local_path
        except Exception as exn:
            exn.args = (*exn.args, filesample.get("stream"), filesample.get("url"))
            if handler(exn):
                continue
            else:
                break
    if current_sample is not None and valid_sample(current_sample):
        yield current_sample


def _atarfile_expander(
    data: Iterable[dict[str, Any]],
    handler: Callable[[Exception], bool] = wds.warn_and_continue,
    select_files: Optional[Callable[[str], bool]] = None,
    rename_files: Optional[Callable[[str], str]] = None,
    eof_value: Optional[Any] = None,
    group_by_keys_split=_split_path,
    group_by_keys_suffixes=None,
    group_by_keys_lcase=True,
    tar_index_handler: Optional[Callable] = None,
    attached_file_handlers_creator: Optional[Callable[[str], list]] = None,
) -> Iterable[dict[str, Any]]:
    for source in data:
        try:
            url = source["url"]
            _extra = {k: v for k, v in source.items() if k not in ["url", "stream"]}
            attached_file_handlers: list = (
                attached_file_handlers_creator(url)
                if attached_file_handlers_creator is not None
                else []
            )
            files = _expand_tarfile(
                source,
                select_files,
                rename_files,
                eof_value,
                handler=handler,
                tar_index_handler=tar_index_handler,
            )
            for sample in group_by_keys(
                files,
                keys=group_by_keys_split,
                handler=handler,
                suffixes=group_by_keys_suffixes,
                lcase=group_by_keys_lcase,
            ):
                sample.update(_extra)
                for attached_file_handler in attached_file_handlers:
                    sample = attached_file_handler(sample)

                yield sample
        except Exception as exn:
            exn.args = (*exn.args, source.get("stream"), source.get("url"))
            if handler(exn):
                continue
            else:
                break


def _atarfile_to_samples(
    data: Iterable[dict[str, Any]],
    handler: Callable[[Exception], bool] = wds.warn_and_continue,
    select_files: Optional[Callable[[str], bool]] = None,
    rename_files: Optional[Callable[[str], str]] = None,
    eof_value: Optional[Any] = None,
    group_by_keys_split=_split_path,
    group_by_keys_suffixes=None,
    group_by_keys_lcase=True,
    tar_index_handler: Optional[Callable] = None,
    attached_file_handlers_creator: Optional[Callable[[str], list]] = None,
) -> Iterable[dict[str, Any]]:
    streams = wds.tariterators.url_opener(data, handler=handler)

    samples = _atarfile_expander(
        streams,
        handler=handler,
        select_files=select_files,
        rename_files=rename_files,
        eof_value=eof_value,
        group_by_keys_split=group_by_keys_split,
        group_by_keys_suffixes=group_by_keys_suffixes,
        group_by_keys_lcase=group_by_keys_lcase,
        tar_index_handler=tar_index_handler,
        attached_file_handlers_creator=attached_file_handlers_creator,
    )

    return samples


# iterate over files from tar and it's attached files,
# and group them by keys then yield samples
atarfile_to_samples = pipelinefilter(_atarfile_to_samples)


class AttachedFileHandler:
    def __init__(self, attached_file_path: str, key_prefix: Optional[str] = None):
        self.key_prefix = key_prefix
        self.attached_file_path = attached_file_path

    @classmethod
    def from_url(
        cls,
        url: str,
        suffix: str,
        attached_file_root=None,
        remove_tar_suffix=False,
        replace=False,
        origin_replace_path=None,
        key_prefix: Optional[str] = None,
    ):
        attached_file_path = cls._attached_file_path(
            url,
            suffix,
            attached_file_root,
            remove_tar_suffix,
            replace,
            origin_replace_path,
        )
        if not megfile.smart_exists(attached_file_path):
            attached_file_path = cls._attached_file_path(
                url, suffix, attached_file_root, True
            )
            if not megfile.smart_exists(attached_file_path):
                raise FileExistsError(
                    f"`{suffix}` attached file for {url} does not exist. {attached_file_root=} {remove_tar_suffix=}"
                )
        return cls(attached_file_path, key_prefix=key_prefix)

    @staticmethod
    def _attached_file_path(
        url: str,
        suffix: str,
        attached_file_root=None,
        remove_tar_suffix=False,
        replace=False,
        origin_replace_path=None,
    ) -> str:
        url_head, url_tail = os.path.split(url)
        if replace:
            assert origin_replace_path is not None
            url = url.replace(origin_replace_path, attached_file_root)  # type: ignore
            return url + suffix
        else:
            attached_file_root = (
                attached_file_root if attached_file_root is not None else url_head
            )
            attached_file_name = (
                url_tail if not remove_tar_suffix else os.path.splitext(url_tail)[0]
            ) + str(suffix)
            return os.path.join(attached_file_root, attached_file_name)

    @staticmethod
    def extent_sample(sample: dict, sample_extra: dict, key_prefix=None):
        if key_prefix is None:
            sample.update(sample_extra)
        else:
            sample.update({f"{key_prefix}{k}": sample_extra[k] for k in sample_extra})
        return sample

    def index(self, sample: dict) -> dict:
        raise NotImplementedError

    def __call__(self, sample: dict) -> dict:
        sample_extra = self.index(sample)
        return self.extent_sample(sample, sample_extra, key_prefix=self.key_prefix)


class AttachedParquetHandler(AttachedFileHandler):
    def __init__(self, attached_file_path: str, key_prefix: Optional[str] = None):
        super().__init__(attached_file_path, key_prefix)

        import pandas as pd

        try:
            dataframe = pd.read_parquet(wds.gopen(self.attached_file_path))  # type: ignore
            if dataframe.index.duplicated().any():
                dataframe = dataframe[~dataframe.index.duplicated(keep=False)]
            self.data_dict = dataframe.to_dict(orient="index")  # type: ignore
        except ValueError as err:
            raise ValueError(f"{err} {self.attached_file_path}") from err

    def index(self, sample: dict) -> dict:
        __key__ = sample["__key__"]
        return self.data_dict.get(__key__, dict())


class AttachedJsonHandler(AttachedFileHandler):
    def __init__(self, attached_file_path: str, key_prefix: Optional[str] = None):
        super().__init__(attached_file_path, key_prefix)

        import json

        try:
            with wds.gopen(self.attached_file_path) as f:
                self.data_dict = json.load(f)  # type: ignore
        except ValueError as err:
            raise ValueError(f"{err} {self.attached_file_path}") from err

    def index(self, sample: dict) -> dict:
        __key__ = sample["__key__"]
        return self.data_dict.get(__key__, dict())


class AttachedPickleHandler(AttachedFileHandler):
    def __init__(self, attached_file_path: str, key_prefix: Optional[str] = None):
        super().__init__(attached_file_path, key_prefix)

        import pickle

        try:
            with wds.gopen(self.attached_file_path) as f:
                self.data_dict = pickle.load(f)  # type: ignore
        except ValueError as err:
            raise ValueError(f"{err} {self.attached_file_path}") from err

    def index(self, sample: dict) -> dict:
        __key__ = sample["__key__"]
        return self.data_dict.get(__key__, dict())


class AttachedFileHandlersCreator:
    def __init__(self, *handler_rules, error_handler=wds.reraise_exception):
        self.error_handler = error_handler
        self._handlers_creator_factory = {}

        for rule in handler_rules:
            handler_cls = None
            if isinstance(rule, str):
                _suffix = rule
                _kwargs = dict()
            elif isinstance(rule, Sequence) and len(rule) == 2:
                _suffix = rule[0]
                _key_prefix = rule[1]
                _kwargs = dict(key_prefix=_key_prefix)
            elif isinstance(rule, dict):
                rule = rule.copy()
                _suffix = rule.pop("suffix")
                handler_cls = rule.pop("_type", None)
                _kwargs = rule
            else:
                raise ValueError(f"can not identify handler for {rule=}")

            _suffix: str
            if handler_cls is None:
                if _suffix.endswith(".parquet"):
                    handler_cls = AttachedParquetHandler
                elif _suffix.endswith(".pickle"):
                    handler_cls = AttachedPickleHandler
                elif _suffix.endswith(".json"):
                    handler_cls = AttachedJsonHandler
                else:
                    raise ValueError(f"can not identify handler for {rule=}")

            self._handlers_creator_factory[_suffix] = (handler_cls, _kwargs)

    def __call__(self, url) -> list[Callable]:
        handlers = []
        for suffix, (handler_cls, kwargs) in self._handlers_creator_factory.items():
            try:
                handler = handler_cls.from_url(url, suffix, **kwargs)
                handlers.append(handler)
            except Exception as exn:
                if self.error_handler(exn):
                    continue
                break
        return handlers


class Selector:
    def __init__(
        self,
        predicate: Callable,
        handler=wds.reraise_exception,
        *args,
        report_when_pass=None,
        **kwargs,
    ) -> None:
        self.predicate = predicate
        self.predicate_args = args
        self.predicate_kwargs = kwargs
        self.handler = handler

        self.num_seen_samples = 0
        self.num_passed_samples = 0
        self.report_when_pass = report_when_pass

    def report(self):
        print(
            f"{distributed_identifier()}"
            f"{self.predicate}(args={self.predicate_args}, kwargs={self.predicate_kwargs})"
            f" passed {self.num_passed_samples}/{self.num_seen_samples}"
            f" = {self.num_passed_samples * 100 / self.num_seen_samples:.2f}%",
        )

    def __call__(self, data: Iterator) -> Any:
        for sample in data:
            self.num_seen_samples += 1
            try:
                if self.predicate(
                    sample, *self.predicate_args, **self.predicate_kwargs
                ):
                    yield sample
                    self.num_passed_samples += 1

                    if (
                        self.report_when_pass is not None
                        and self.num_passed_samples % self.report_when_pass == 0
                    ):
                        self.report()
            except Exception as exn:
                _add_note_to_exception(
                    exn,
                    f"{self.predicate}(args={self.predicate_args}, kwargs={self.predicate_kwargs})",
                )
                if self.handler(exn):
                    continue
                else:
                    break


def select(
    *args, predicate: Callable, handler=wds.reraise_exception, **kwargs
) -> Selector:
    return Selector(predicate, handler, *args, **kwargs)


def _must_in_range(sample, rules, allow_not_exist=False):
    for key, rule in rules.items():
        if len(rule) == 2:
            v_min, v_max = rule
            _allow_not_exist = allow_not_exist
        elif len(rule) == 3:
            v_min, v_max, _allow_not_exist = rule
        else:
            raise ValueError(f"invalid rule for {key}: {rule}")

        if key not in sample:
            if _allow_not_exist:
                continue
            return False

        v = sample[key]

        if (v_min is not None and v < v_min) or (v_max is not None and v > v_max):
            return False

    return True


must_in_range = partial(select, predicate=_must_in_range)


def no_op(x):
    return x


def _must_exist_keys(sample: dict, keys: Union[str, list[str]]):
    if isinstance(keys, str):
        keys = [keys]
    return all(k in sample for k in keys)


must_exist_keys = partial(select, predicate=_must_exist_keys)


def _switch_pipelines(data, router: Callable | None = None, **pipelines):
    for sample in data:
        routing_key = (
            sample["__source__"]
            if isinstance(sample, dict) and "__source__" in sample
            else (router(sample) if router is not None else "__default__")
        )

        pipeline_stage = pipelines.get(routing_key, pipelines.get("__default__"))

        if pipeline_stage is None:
            raise ValueError(
                f"can not find routing key {routing_key} in {pipelines.keys()}"
            )

        yield from pipeline_stage([sample])


switch_pipelines = pipelinefilter(_switch_pipelines)


def _map_sample(data, f, handler=wds.reraise_exception, *args, **kwargs):
    """Map samples."""
    for sample in data:
        try:
            result = f(sample, *args, **kwargs)
        except Exception as exn:
            if handler(exn):
                continue
            else:
                break
        if result is None:
            continue
        if isinstance(sample, dict) and isinstance(result, dict):
            result["__key__"] = sample.get("__key__")
        yield result


map_sample = pipelinefilter(_map_sample)


def _map_dict(data, handler=wds.reraise_exception, **kw):
    """
    Map the entries in a dict sample with individual functions.

    Args:
        data: Source iterator of dictionary samples.
        handler: Exception handler function.
        **kw: Mapping of keys to functions to apply.

    Yields:
        Samples with mapped values.

    Raises:
        Exception: If the handler doesn't handle an exception.
    """
    assert len(list(kw.keys())) > 0
    for key, f in kw.items():
        assert callable(f), (key, f)

    for sample in data:
        assert isinstance(sample, dict)
        try:
            for k, f in kw.items():
                sample[k] = f(sample[k])
        except Exception as exn:
            __key__ = sample.get("__key__", "N/A")
            __url__ = sample.get("__url__", "N/A")
            _add_note_to_exception(exn, f"{__key__=} {__url__=}")
            if handler(exn):
                continue
            else:
                break
        yield sample


map_dict = pipelinefilter(_map_dict)


def _decode_image(sample: dict, imagespec="pil", image_key="image"):
    sample[image_key] = wds.autodecode.ImageHandler(imagespec, extensions=[image_key])(
        image_key, data=sample[image_key]
    )
    return sample


decode_image = pipelinefilter(partial(_map_sample, f=_decode_image))


def _assign_key(sample, handler=print_exception_and_continue, **assigners):
    try:
        for key, assigner in assigners.items():
            sample[key] = assigner(sample)
        return sample
    except Exception as exn:
        if handler(exn):
            return sample


assign_key = pipelinefilter(partial(_map_sample, f=_assign_key))


def _only_keep(sample, keys, keep_meta=True):
    _keep_keys = {*keys, "__key__", "__url__"} if keep_meta else set(keys)
    return {k: sample[k] for k in _keep_keys if k in sample}


only_keep = pipelinefilter(partial(_map_sample, f=_only_keep))


def _info(data, fmt=None, n=3, every=-1, width=120, stream=sys.stderr, name=""):
    name = f"{name}{distributed_identifier()}"

    for i, sample in enumerate(data):
        if i < n or (every > 0 and (i + 1) % every == 0):
            if fmt is None:
                print("---", name, file=stream)
                for k, v in sample.items():
                    print(k, repr(v)[:width], file=stream)
            else:
                print(fmt.format(**sample), file=stream)
        yield sample


info = pipelinefilter(_info)


def _iterate_from_sample(
    data: Iterable, iterator_from: Callable, handler=wds.reraise_exception
):
    for sample in data:
        try:
            yield from iterator_from(sample)
        except Exception as exn:
            if handler(exn):
                continue
            else:
                break


iterate_from_sample = pipelinefilter(_iterate_from_sample)


def _iterate_from_mix_samples(
    data: Iterable, iterator_from: Callable, n=2, handler=wds.reraise_exception
):
    if n < 1:
        raise ValueError("n must be at least one")

    batched_sample = []

    for sample in data:
        try:
            batched_sample.append(sample)

            if len(batched_sample) < n:
                continue

            yield from wds.mix.random_samples(
                [iterator_from(s) for s in batched_sample],
                probs=None,
                longest=True,
            )
            batched_sample = []
        except Exception as exn:
            if handler(exn):
                continue
            else:
                break
    try:
        yield from wds.mix.random_samples(
            [iterator_from(s) for s in batched_sample],
            probs=None,
            longest=True,
        )
    except Exception as exn:
        handler(exn)


iterate_from_mix_samples = pipelinefilter(_iterate_from_mix_samples)


def _group_by(
    data: Iterable,
    sample_to_group_id: Optional[Callable[[dict, Callable], str]] = None,
    handler=wds.reraise_exception,
    num_max_size=None,
):
    """
    Group samples in the given iterable by the specified strategy.

    Args:
        data: The iterable of samples.
        sample_to_group_id: A callable that takes a sample and returns a group id.
        handler: An exception handler.
        num_max_size: The maximum number of samples in a group.

    Yields:
        A generator of grouped samples.
    """
    current_group_samples = []
    current_group_id = None

    for sample in data:
        try:
            # get the group id of the current sample
            if sample_to_group_id is None:
                assert isinstance(sample, dict)
                group_id = sample.get("__group__", sample["__key__"])
            else:
                group_id = sample_to_group_id(sample, handler)

            # if the group id of the current sample is the same as the previous sample,
            # add it to the current group
            if (current_group_id is None or current_group_id == group_id) and (
                num_max_size is None or len(current_group_samples) < num_max_size
            ):
                current_group_id = group_id
                current_group_samples.append(sample)

                continue

            if len(current_group_samples):
                yield current_group_samples

            current_group_samples = [sample]
            current_group_id = group_id
        except Exception as exn:
            # if an exception occurs, call the handler
            if handler(exn):
                continue
            else:
                break

    # if there are remaining samples in the last group, yield them
    if len(current_group_samples):
        yield current_group_samples


group_by = pipelinefilter(_group_by)
