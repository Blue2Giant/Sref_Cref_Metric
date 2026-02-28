CREATE TABLE IF NOT EXISTS sequences_data (
    sequence_id UUID PRIMARY KEY,
    vault_path VARCHAR,
    create_time timestamp,
    uri VARCHAR,
    source VARCHAR,
    task_type VARCHAR,
    sequence_choices STRUCT(
        choice STRUCT(
            type VARCHAR,
            index VARCHAR,
            require_loss BOOLEAN,
            ref_index VARCHAR
        )[]
    )[],
    choices_weights DOUBLE[],
    images STRUCT(
        id UUID,
        width INTEGER,
        height INTEGER,
        index VARCHAR
    )[],
    texts STRUCT(
        id UUID,
        content VARCHAR,
        language VARCHAR(10),
        index VARCHAR
    )[]
);