"""Evaluation helpers for recall@k."""

from __future__ import annotations

from .models import AnsweredQuestion, MinimalSource, StudentSearchResults


def compute_recall_at_k(
    student_results: StudentSearchResults,
    answered_questions: list[AnsweredQuestion],
    k: int,
) -> float:
    """Compute recall@k using file match plus span overlap."""

    if k <= 0 or not answered_questions:
        return 0.0

    answered_by_id = {question.question_id: question for question in answered_questions}
    total_ratio = 0.0
    question_count = 0

    for result in student_results.search_results:
        ground_truth = answered_by_id.get(result.question_id)
        if ground_truth is None or not ground_truth.sources:
            continue
        hits = 0
        for expected_source in ground_truth.sources:
            if _has_matching_source(expected_source, result.retrieved_sources[:k]):
                hits += 1
        total_ratio += hits / len(ground_truth.sources)
        question_count += 1

    if question_count == 0:
        return 0.0
    return total_ratio / question_count


def _has_matching_source(
    expected_source: MinimalSource,
    retrieved_sources: list[MinimalSource],
) -> bool:
    """Check whether one retrieved source overlaps the expected source enough."""

    for source in retrieved_sources:
        if source.file_path != expected_source.file_path:
            continue
        if _intersection_over_union(source, expected_source) >= 0.05:
            return True
    return False


def _intersection_over_union(left: MinimalSource, right: MinimalSource) -> float:
    """Compute one-dimensional intersection over union."""

    intersection_start = max(left.first_character_index, right.first_character_index)
    intersection_end = min(left.last_character_index, right.last_character_index)
    intersection = max(0, intersection_end - intersection_start)
    if intersection == 0:
        return 0.0
    union_start = min(left.first_character_index, right.first_character_index)
    union_end = max(left.last_character_index, right.last_character_index)
    union = union_end - union_start
    if union <= 0:
        return 0.0
    return intersection / union
