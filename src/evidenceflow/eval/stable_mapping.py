"""Independently validate the Source locator, then map the selected occurrence onto the
read-only Corpus."""

import re
import unicodedata

from evidenceflow.domain.evaluation_v2 import GroupMapping, MappingArtifact


NORMALIZATION_VERSION = "source-guided-hyphen-v1"


def hyphen_rules(raw_source: str, anchor: str) -> list[dict[str, str]]:
    """Merge a concrete hyphenated line break only when both the original page break and the
    complete Gold word prove it."""
    source = unicodedata.normalize("NFKC", raw_source)
    anchor = unicodedata.normalize("NFKC", anchor).casefold()
    rules = []
    for match in re.finditer(r"\b([A-Za-z]+)-[ \t]*\r?\n\s*([A-Za-z]+)\b", source):
        left, right = (part.casefold() for part in match.groups())
        joined = left + right
        # Exclude the components of legitimate hyphenated compounds; the repair is never
        # inferred from a dictionary or from the Retriever.
        if re.search(r"(?<![\w-])" + re.escape(joined) + r"(?![\w-])", anchor):
            rule = {"left": left, "right": right, "joined": joined}
            if rule not in rules:
                rules.append(rule)
        elif left + "-" + right in anchor:
            # When both the source and the anchor prove a legitimate hyphen, only the line
            # break is repaired and the hyphen itself is kept.
            rule = {"left": left, "right": right, "joined": left + "-" + right}
            if rule not in rules:
                rules.append(rule)
    return rules


def normalize_mapping(text: str, rules: list[dict[str, str]]) -> str:
    """Keep the original string and return a normalized copy that applies only the
    source-proven rules."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    for rule in rules:
        pattern = r"\b" + re.escape(rule["left"]) + r"-\s+" + re.escape(rule["right"]) + r"\b"
        normalized = re.sub(pattern, rule["joined"], normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _positions(text, anchor):
    """Return every overlapping occurrence position, preserving the original occurrence
    numbering."""
    return [m.start() for m in re.finditer(f"(?={re.escape(anchor)})", text)] if anchor else []


def validate_source(alternative, document) -> dict:
    """Consume only the original document and the Canonical Gold, fully independent of
    Chunks and retrieval results."""
    result = {"source_valid": False, "source_locator_unique": False, "source_match_count": 0,
              "normalization_used": NORMALIZATION_VERSION, "notes": [], "rules": []}
    if document is None or document.content_hash != alternative.content_version:
        result["notes"] = ["source absent or content hash mismatch"]
        return result
    loc = alternative.locator
    if loc.page_start is not None:
        selected = [s for s in document.segments if s.page_number is not None and loc.page_start <= s.page_number <= loc.page_end]
        if {s.page_number for s in selected} != set(range(loc.page_start, loc.page_end + 1)):
            result["notes"] = ["page locator mismatch"]
            return result
        raw = "\n".join(s.content for s in selected)
    else:
        if loc.block_index >= len(document.segments):
            result["notes"] = ["block locator mismatch"]
            return result
        raw = document.segments[loc.block_index].content
        if loc.span_start is not None:
            if loc.span_end > len(raw):
                result["notes"] = ["block span exceeds source"]
                return result
            raw = raw[loc.span_start:loc.span_end]
    rules = hyphen_rules(raw, alternative.evidence_anchor)
    text = normalize_mapping(raw, rules)
    anchor = normalize_mapping(alternative.evidence_anchor, rules)
    positions = _positions(text, anchor)
    matches = []
    for occurrence, start in enumerate(positions, 1):
        end = start + len(anchor)
        if loc.occurrence is not None and occurrence != loc.occurrence:
            continue
        if loc.left_context and not text[:start].rstrip().endswith(normalize_mapping(loc.left_context, rules)):
            continue
        if loc.right_context and not text[end:].lstrip().startswith(normalize_mapping(loc.right_context, rules)):
            continue
        matches.append((start, end, occurrence))
    result.update(raw_source_text=raw, normalized_source_text=text,
                  raw_anchor=alternative.evidence_anchor, normalized_anchor=anchor,
                  rules=rules, source_occurrence_count=len(positions), source_match_count=len(matches))
    if len(matches) != 1:
        result["notes"] = ["locator must resolve exactly one occurrence"]
        return result
    start, end, occurrence = matches[0]
    result.update(source_valid=True, source_locator_unique=True, source_span_start=start,
                  source_span_end=end, selected_occurrence=occurrence)
    if loc.source_span_start is not None and (start, end) != (loc.source_span_start, loc.source_span_end):
        result["notes"].append("auxiliary offset drift; context/occurrence remain authoritative")
    return result


def _compact(text):
    """Ignore whitespace for position alignment, keeping all characters, digits and
    punctuation."""
    return re.sub(r"\s+", "", text)


def map_source_to_chunks(alternative, document, chunks, validation):
    """Align the selected position; coverage can never be obtained through another
    same-named anchor on the same page."""
    if not validation["source_valid"]:
        raise ValueError("source validation must pass before corpus mapping")
    rules = validation["rules"]
    loc = alternative.locator
    normalized = validation["normalized_source_text"]
    begin, finish = validation["source_span_start"], validation["source_span_end"]
    # The scope position in the full document is fixed by the original page/block, never by
    # Chunks or retrieval preference.
    segments = document.segments
    first = next(i for i, s in enumerate(segments) if s.page_number == loc.page_start) if loc.page_start is not None else loc.block_index
    prefix = "".join(_compact(normalize_mapping(s.content, rules)) for s in segments[:first])
    if loc.span_start is not None:
        prefix += _compact(normalize_mapping(segments[first].content[:loc.span_start], rules))
    full_source = "".join(_compact(normalize_mapping(s.content, rules)) for s in segments)
    start = len(prefix) + len(_compact(normalized[:begin]))
    end = len(prefix) + len(_compact(normalized[:finish]))
    # A proof with context is the conservative fallback when whole-chunk alignment fails, so
    # a short anchor alone is not enough.
    left = normalize_mapping(loc.left_context, rules) if loc.left_context else ""
    right = normalize_mapping(loc.right_context, rules) if loc.right_context else ""
    proof = _compact(left + normalized[begin:finish] + right)
    proof_unique = len(_positions(full_source, proof)) == 1
    intervals, trace = [], []
    for chunk in sorted(chunks, key=lambda c: c.chunk_id):
        if chunk.document_id != document.document_id:
            continue
        if chunk.content_hash != alternative.content_version:
            raise ValueError("chunk/source content version mismatch")
        if loc.page_start is not None and (chunk.page_start is None or chunk.page_end is None
                or chunk.page_start > loc.page_end or chunk.page_end < loc.page_start):
            continue
        content = normalize_mapping(chunk.content, rules)
        compact = _compact(content)
        positions = _positions(full_source, compact)
        relevant = [(p, p + len(compact), chunk.chunk_id) for p in positions
                    if p < end and p + len(compact) > start]
        # Only a unique whole-chunk alignment may prove the fragment through its interval;
        # overlapping chunks are distinct legitimate alternatives.
        if len(positions) == 1:
            intervals.extend(relevant)
        elif proof_unique and proof and proof in compact:
            intervals.append((start, end, chunk.chunk_id))
        else:
            continue
        if relevant or (proof_unique and proof in compact):
            trace.append({"chunk_id": chunk.chunk_id, "raw_text": chunk.content,
                          "normalized_text": content, "rules": rules,
                          "alignment": "whole_chunk" if len(positions) == 1 else "unique_context_proof"})
    complete = [[cid] for a, b, cid in intervals if a <= start and b >= end]
    if complete:
        return "FULLY_MAPPED", complete, trace
    cursor, bundle = start, []
    while cursor < end:
        options = [item for item in intervals if item[0] <= cursor < item[1]]
        if not options:
            break
        chosen = max(options, key=lambda item: (item[1], item[2]))
        cursor = chosen[1]
        bundle.append(chosen[2])
    if cursor >= end and bundle:
        return "FRAGMENTED_ACROSS_CHUNKS", [bundle], trace
    return ("PARTIALLY_MAPPED" if intervals else "UNMAPPED"), [], trace


def map_stable_dataset(dataset, documents, chunks, *, index_version):
    """Complete every Source Gate first, then derive the Corpus status and the traceable text
    evidence."""
    by_id = {d.document_id: d for d in documents}
    if len(by_id) != len(documents) or len({c.chunk_id for c in chunks}) != len(chunks):
        raise ValueError("duplicate document/chunk IDs")
    validations = []
    for q in dataset.questions:
        for group in q.required_evidence_groups:
            for index, alternative in enumerate(group.alternatives):
                result = validate_source(alternative, by_id.get(alternative.document_id))
                validations.append({"question_id": q.question_id, "group_id": group.group_id,
                                    "alternative_index": index, **result})
    if any(not row["source_valid"] for row in validations):
        return None, validations, []
    rows, traces = [], []
    for q in dataset.questions:
        for group in q.required_evidence_groups:
            outcomes = []
            for index, alternative in enumerate(group.alternatives):
                validation = next(v for v in validations if (v["question_id"], v["group_id"], v["alternative_index"]) == (q.question_id, group.group_id, index))
                status, bundles, trace = map_source_to_chunks(alternative, by_id[alternative.document_id], chunks, validation)
                outcomes.append((status, bundles))
                traces.append({"question_id": q.question_id, "group_id": group.group_id,
                               "alternative_index": index, "status": status, "chunks": trace})
            bundles = [bundle for _, sets in outcomes for bundle in sets]
            status = "FULLY_MAPPED" if any(len(b) == 1 for b in bundles) else (
                "FRAGMENTED_ACROSS_CHUNKS" if bundles else (
                    "PARTIALLY_MAPPED" if any(s == "PARTIALLY_MAPPED" for s, _ in outcomes) else "UNMAPPED"))
            rows.append(GroupMapping(question_id=q.question_id, group_id=group.group_id, status=status,
                                     chunk_sets=bundles, reason="unique source locator; source-position-based corpus alignment"))
    return MappingArtifact(mapping_version="gold-mapper-v3", normalization_version=NORMALIZATION_VERSION,
                           dataset_id=dataset.dataset_id, corpus_version=dataset.corpus_version,
                           index_version=index_version, groups=rows), validations, traces
