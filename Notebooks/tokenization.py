def tokenize_fun(df, tokenizer, max_seq_length=384):
    questions = [q.strip() for q in df["question"]]
    passages = [p.strip() for p in df['passage']]

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    inputs = tokenizer(
        questions,
        passages,
        max_length=max_seq_length,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length"
    )

    offset_mapping = inputs.pop("offset_mapping")
    answers = df["answers"]

    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        answer = answers[i]
        start_char = answer[0]["start_char"]
        end_char = start_char + len(answer[0]["text"])
        sequence_ids = inputs.sequence_ids(i)

        passage_start = sequence_ids.index(1)
        passage_end = passage_start
        while passage_end < len(sequence_ids) and sequence_ids[passage_end] == 1:
            passage_end += 1
        passage_end -= 1

        if offset[passage_start][0] > end_char or offset[passage_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            idx = passage_start
            while idx <= passage_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = passage_end
            while idx >= passage_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs