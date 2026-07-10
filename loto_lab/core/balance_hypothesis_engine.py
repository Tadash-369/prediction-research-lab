from __future__ import annotations

from collections import Counter
from statistics import mean


BALANCE_HYPOTHESIS_KEY = "balance_hypothesis_engine"
BALANCE_HYPOTHESIS_LABEL = "バランス仮説エンジン"


def _clean_rows(rows, number_max):
    cleaned = []
    for row in rows or []:
        values = set()
        for number in row:
            try:
                value = int(number)
            except (TypeError, ValueError):
                continue
            if 1 <= value <= number_max:
                values.add(value)
        numbers = sorted(values)
        if numbers:
            cleaned.append(numbers)
    return cleaned


def _percentile(values, ratio):
    values = sorted(float(value) for value in values)
    if not values:
        return 0.0
    index = min(len(values) - 1, max(0, int(round((len(values) - 1) * ratio))))
    return values[index]


def _longest_run(numbers):
    longest = 1
    current = 1
    ordered = sorted(numbers)
    for previous, current_number in zip(ordered, ordered[1:]):
        if current_number == previous + 1:
            current += 1
            longest = max(longest, current)
        else:
            current = 1
    return longest if ordered else 0


def _consecutive_pairs(numbers):
    ordered = sorted(numbers)
    return sum(1 for previous, current in zip(ordered, ordered[1:]) if current == previous + 1)


def _last_seen_gaps(rows, number_max):
    gaps = {}
    for number in range(1, number_max + 1):
        gap = len(rows)
        for offset, row in enumerate(reversed(rows), start=1):
            if number in row:
                gap = offset
                break
        gaps[number] = gap
    return gaps


def _grade(score, warnings):
    if score >= 82 and not warnings:
        return "良好"
    if score >= 68:
        return "やや偏り"
    if score >= 52:
        return "偏り強め"
    return "過去傾向外"


def evaluate_balance_hypothesis(
    numbers,
    history_rows,
    *,
    number_max,
    draw_size,
    bonus_rows=None,
    set_ball_analysis=None,
):
    selected = sorted({int(number) for number in numbers if 1 <= int(number) <= number_max})
    rows = _clean_rows(history_rows, number_max)
    bonus_rows = _clean_rows(bonus_rows or [], number_max)
    reasons = []
    warnings = []
    score = 100.0

    if len(selected) != draw_size:
        return {
            "balance_score": 0.0,
            "balance_grade": "過去傾向外",
            "balance_reasons": [],
            "balance_warnings": [f"{draw_size}個の重複なし数字ではありません"],
        }

    odd_count = sum(number % 2 for number in selected)
    low_limit = 22 if draw_size == 6 else 18
    low_count = sum(number <= low_limit for number in selected)

    if draw_size == 6:
        if odd_count == 3:
            reasons.append("奇数偶数バランスが3:3で安定")
        elif odd_count in (2, 4):
            score -= 6
            reasons.append("奇数偶数バランスは許容範囲")
        elif odd_count in (1, 5):
            score -= 17
            warnings.append("奇数偶数の偏りが強め")
        else:
            score -= 28
            warnings.append("奇数偶数が極端に偏っています")
    else:
        if odd_count in (3, 4):
            reasons.append("奇数偶数バランスが3:4または4:3で安定")
        elif odd_count in (2, 5):
            score -= 7
            reasons.append("奇数偶数バランスは許容範囲")
        elif odd_count in (1, 6):
            score -= 18
            warnings.append("奇数偶数の偏りが強め")
        else:
            score -= 30
            warnings.append("奇数偶数が極端に偏っています")

    ideal_low_counts = (3,) if draw_size == 6 else (3, 4)
    if low_count in ideal_low_counts:
        reasons.append("高低バランスが安定")
    elif low_count in (2, 4, 5):
        score -= 7
        reasons.append("高低バランスは許容範囲")
    elif low_count in (1, draw_size - 1):
        score -= 18
        warnings.append("高低の偏りが強め")
    else:
        score -= 30
        warnings.append("高低が極端に偏っています")

    if rows:
        sums = [sum(row) for row in rows if len(row) == draw_size]
        if sums:
            total = sum(selected)
            center = mean(sums)
            lower = _percentile(sums, 0.10)
            upper = _percentile(sums, 0.90)
            distance = abs(total - center)
            if lower <= total <= upper:
                reasons.append("合計値が過去分布の中心帯に近い")
            else:
                score -= min(22, max(8, distance / max(draw_size, 1)))
                warnings.append("合計値が過去分布から外れ気味")

        frequency = Counter(number for row in rows for number in row)
        recent_rows = rows[-20:]
        recent_frequency = Counter(number for row in recent_rows for number in row)
        selected_recent_hot = sum(1 for number in selected if recent_frequency[number] >= max(2, len(recent_rows) // 8))
        selected_long_cold = sum(1 for number in selected if frequency[number] <= max(1, len(rows) // max(number_max, 1)))
        if selected_recent_hot and selected_long_cold:
            reasons.append("ホット数字とコールド数字が混在")
        elif selected_recent_hot == 0:
            score -= 7
            warnings.append("直近傾向要素が少なめ")
        elif selected_long_cold == 0:
            score -= 5
            warnings.append("コールド要素が少なめ")

        gaps = _last_seen_gaps(rows, number_max)
        short_gap = sum(1 for number in selected if gaps.get(number, 0) <= 3)
        long_gap = sum(1 for number in selected if gaps.get(number, 0) >= max(8, len(rows) // 10))
        if short_gap and long_gap:
            reasons.append("出現間隔に短期・長期が混在")
        elif short_gap == draw_size or long_gap == draw_size:
            score -= 12
            warnings.append("出現間隔が片側に寄っています")

    longest_run = _longest_run(selected)
    consecutive_pairs = _consecutive_pairs(selected)
    if longest_run <= 2 and consecutive_pairs <= 1:
        reasons.append("連番の出方が穏やか")
    elif longest_run == 3 or consecutive_pairs == 2:
        score -= 8
        warnings.append("連番がやや多め")
    elif longest_run >= 4 or consecutive_pairs >= 3:
        score -= 22
        warnings.append("連番が多すぎます")

    last_digits = Counter(number % 10 for number in selected)
    if max(last_digits.values()) >= 4:
        score -= 14
        warnings.append("下一桁が集中")
    elif max(last_digits.values()) <= 2:
        reasons.append("下一桁の偏りが少ない")

    tens_bands = Counter(number // 10 for number in selected)
    if max(tens_bands.values()) >= 4:
        score -= 12
        warnings.append("十の位が集中")
    elif len(tens_bands) >= min(3, draw_size):
        reasons.append("十の位の分散がある")

    if bonus_rows:
        latest_bonus = set(bonus_rows[-1])
        near_bonus = [
            number
            for number in selected
            if any(abs(number - bonus) <= 3 for bonus in latest_bonus)
        ]
        if near_bonus:
            score += min(4, len(near_bonus))
            reasons.append("ボーナス数字周辺を軽く反映")

    if set_ball_analysis and set_ball_analysis.get("available"):
        score += 3
        reasons.append("セット球分析を補助要素として反映")

    score = max(0.0, min(100.0, score))
    return {
        "balance_score": round(score, 3),
        "balance_grade": _grade(score, warnings),
        "balance_reasons": reasons[:8],
        "balance_warnings": warnings[:8],
    }
