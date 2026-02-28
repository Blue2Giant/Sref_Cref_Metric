import io
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Sequence

import PIL.Image
import PIL.ImageOps

from vault.schema import ID
from vault.utils import jsonify_meta


class MultiModalType(str, Enum):
    IMAGE = "image"
    TEXT = "text"
    SAMPLE_ANNOTATION = "sample_annotation"


@dataclass(frozen=True)
class Creator:
    id: ID
    name: str
    meta: Any | None
    json_meta: str | None = field(repr=False, default=None)

    @classmethod
    def create(cls, name: str, meta: Any | None = None):
        json_meta = jsonify_meta(meta)
        return cls(
            id=ID.hash(name, json_meta), name=name, meta=meta, json_meta=json_meta
        )


@dataclass(frozen=True)
class Annotation:
    id: ID
    name: str
    type_: str | None
    blob: bytes | None = None
    creator: Creator | None = None
    meta: Any | None = None
    json_meta: str | None = field(repr=False, default=None)

    @classmethod
    def create(
        cls,
        name: str,
        type_: str | None,
        blob: bytes | None = None,
        creator: Creator | None = None,
        meta: Any | None = None,
    ):
        json_meta = jsonify_meta(meta)
        return cls(
            id=ID.hash(name, type_, json_meta, blob),
            name=name,
            type_=type_,
            meta=meta,
            blob=blob,
            creator=creator,
        )

    @classmethod
    def generated_by(cls, model: str, generation_config: Any = None) -> "Annotation":
        return cls.create(name=model, type_="generated_by", meta=generation_config)

    @classmethod
    def image_type(cls, name: str) -> "Annotation":
        return cls.create(name=name, type_="image_type")

    @classmethod
    def text_type(cls, name: str) -> "Annotation":
        return cls.create(name=name, type_="text_type")

    @classmethod
    def sequence_type(cls, name: str) -> "Annotation":
        return cls.create(name=name, type_="sequence_type")


@dataclass(frozen=True)
class SampleAnnotation:
    id: ID
    name: str
    sequence_id: ID
    creator: Creator
    value: Any
    participants: tuple[tuple[ID, MultiModalType, str], ...]

    @classmethod
    def create(
        cls,
        name: str,
        sequence_id: ID,
        creator: Creator,
        value: Any,
        participants: tuple[tuple[ID, MultiModalType, str], ...],
    ):
        return cls(
            id=ID.hash(name, creator.id, participants),
            name=name,
            sequence_id=sequence_id,
            creator=creator,
            value=value,
            participants=participants,
        )


@dataclass(frozen=True)
class Image:
    id: ID
    uri: str
    source: str
    pil_image: PIL.Image.Image = field(repr=False)
    blob: bytes = field(repr=False)
    annotations: list[Annotation] | None = None

    @classmethod
    def create(
        cls,
        image: bytes | PIL.Image.Image,
        uri: str,
        source: str,
        annotations: list[Annotation] | None = None,
    ):
        if isinstance(image, bytes):
            image_bytes = image
            pil_image = PIL.Image.open(io.BytesIO(image_bytes))
            pil_image.load()
        elif isinstance(image, PIL.Image.Image):
            buffer = io.BytesIO()
            image.save(buffer, format="webp", lossless=True)
            image_bytes = buffer.getvalue()
            pil_image = image
        else:
            raise ValueError(f"invalid image type {type(image)}")

        if hasattr(pil_image, "_getexif") and pil_image._getexif() is not None:  # type: ignore
            pil_image = PIL.ImageOps.exif_transpose(pil_image)

        return cls(
            id=ID.hash(source, uri, image_bytes),
            pil_image=pil_image,
            blob=image_bytes,
            uri=uri,
            source=source,
            annotations=annotations,
        )


@dataclass(frozen=True)
class Text:
    id: ID
    content: str
    uri: str | None
    source: str | None
    language: str | None
    annotations: list[Annotation] | None = None

    @classmethod
    def create(
        cls,
        content: str,
        uri: str | None,
        source: str | None,
        language: str | None = None,
        annotations: list[Annotation] | None = None,
    ):
        return cls(
            id=ID.hash(source, uri, content),
            content=content,
            uri=uri,
            source=source,
            language=language,
            annotations=annotations,
        )


@dataclass(frozen=True)
class PackSequence:
    id: ID
    images: Sequence[tuple[Image, str | int]]
    texts: Sequence[tuple[Text, str | int]]
    source: str
    uri: str
    annotations: list[Annotation] | None = None
    meta: Any | None = None
    json_meta: str | None = field(repr=False, default=None)

    @classmethod
    def create(
        cls,
        images: Sequence[tuple[Image, str | int]],
        texts: Sequence[tuple[Text, str | int]],
        source: str,
        uri: str,
        annotations: list[Annotation] | None = None,
        meta: Any | None = None,
        source_uri_as_id: bool = True,
    ):
        json_meta = jsonify_meta(meta)

        if source_uri_as_id:
            _id = ID.hash(source, uri)
        else:
            item_ids = [img.id for img, _ in images] + [txt.id for txt, _ in texts]
            _id = ID.hash(source, uri, item_ids)

        return cls(
            id=_id,
            images=images,
            texts=texts,
            source=source,
            uri=uri,
            annotations=annotations,
            meta=meta,
            json_meta=json_meta,
        )

    @classmethod
    def from_sequence(
        cls,
        sequence: Sequence[Image | Text | Sequence[Text] | Sequence[Image]],
        source: str,
        uri: str,
        annotations: list[Annotation] | None = None,
        meta: Any | None = None,
        source_uri_as_id: bool = False,
    ):
        images = []
        texts = []

        for i, item in enumerate(sequence):
            if isinstance(item, Image):
                images.append((item, i))
            elif isinstance(item, Text):
                texts.append((item, i))
            elif isinstance(item, Sequence):
                assert len(item) > 0, f"{i=} {item=} must have more than 1 item"
                if isinstance(item[0], Image):
                    for img in item:
                        assert isinstance(img, Image)
                        images.append((img, i))
                elif isinstance(item[0], Text):
                    for txt in item:
                        assert isinstance(txt, Text)
                        texts.append((txt, i))
                else:
                    raise ValueError(f"invalid sequence item:  {type(item)=} {item=}")
            else:
                raise ValueError(f"invalid sequence item:  {type(item)=} {item=}")

        return cls.create(
            images=images,
            texts=texts,
            source=source,
            annotations=annotations,
            uri=uri,
            meta=meta,
            source_uri_as_id=source_uri_as_id,
        )

    @classmethod
    def from_text_to_image(
        cls,
        caption: Text | Sequence[Text],
        image: Image,
        source: str,
        uri: str,
        annotations: list[Annotation] | None = None,
        meta: Any | None = None,
        source_uri_as_id: bool = False,
    ):
        if annotations is None:
            annotations = [Annotation.sequence_type(name="text_to_image")]

        if isinstance(caption, Text):
            caption = [caption]

        return cls.create(
            images=[(image, "image")],
            texts=[(c, "caption") for c in caption],
            source=source,
            annotations=annotations,
            uri=uri,
            meta=meta,
            source_uri_as_id=source_uri_as_id,
        )

    @classmethod
    def from_t2i_reward(
        cls,
        caption: Text,
        image: Image | Sequence[Image],
        source: str,
        uri: str,
        annotations: list[Annotation] | None = None,
        meta: Any | None = None,
    ):
        if annotations is None:
            annotations = [Annotation.sequence_type(name="t2i_reward")]

        if isinstance(image, Image):
            image = [image]

        return cls.create(
            images=[(img, "image") for img in image],
            texts=[(caption, "caption")],
            source=source,
            annotations=annotations,
            uri=uri,
            meta=meta,
            source_uri_as_id=True,
        )


@dataclass(frozen=True)
class PackSequenceIndex:
    sequence_id: ID
    index: int | str
