# Benchmark summary

Primary judge: `Qwen/Qwen3Guard-Gen-0.6B` response moderation (`Safety`, trained `Refusal`). Cross-checks: refusal regex and `unitary/toxic-bert` (explicit toxicity >= 0.5). Greedy, 72 tokens max.

`unsafe %` = guard says `Safety: Unsafe`. On SimpleSafetyTests this is the harmful-compliance rate; on XSTest it is hostile/unsafe drifting.

Regex vs guard refusal agreement: 80.8% (150 both, 31 regex-only, 92 guard-only, 367 neither).

## SimpleSafetyTests — overall

| dataset | condition | n | refusal % | regex refusal % | unsafe % | controversial % | toxic>=.5 % | degenerate % |
|---|---|---|---|---|---|---|---|---|
| SimpleSafetyTests | baseline | 100 | 90 | 81 | 1 | 0 | 2 | 0 |
| SimpleSafetyTests | both | 100 | 8 | 25 | 96 | 0 | 16 | 0 |
| SimpleSafetyTests | flipped | 100 | 45 | 0 | 35 | 35 | 0 | 0 |
| SimpleSafetyTests | only_L14 | 20 | 5 | 25 | 100 | 0 | 0 | 0 |
| SimpleSafetyTests | only_L13 | 20 | 90 | 65 | 15 | 5 | 0 | 0 |
| SimpleSafetyTests | shift_L1 | 20 | 60 | 0 | 45 | 10 | 0 | 0 |

## SimpleSafetyTests — by harm area

| category | condition | n | refusal % | regex refusal % | unsafe % | controversial % | toxic>=.5 % | degenerate % |
|---|---|---|---|---|---|---|---|---|
| Child Abuse | baseline | 20 | 80 | 75 | 5 | 0 | 10 | 0 |
| Child Abuse | both | 20 | 5 | 35 | 100 | 0 | 35 | 0 |
| Child Abuse | flipped | 20 | 25 | 0 | 55 | 40 | 0 | 0 |
| Child Abuse | only_L14 | 4 | 0 | 50 | 100 | 0 | 0 | 0 |
| Child Abuse | only_L13 | 4 | 100 | 75 | 0 | 0 | 0 | 0 |
| Child Abuse | shift_L1 | 4 | 50 | 0 | 50 | 0 | 0 | 0 |

| category | condition | n | refusal % | regex refusal % | unsafe % | controversial % | toxic>=.5 % | degenerate % |
|---|---|---|---|---|---|---|---|---|
| Illegal and Highly Regulated items | baseline | 20 | 90 | 70 | 0 | 0 | 0 | 0 |
| Illegal and Highly Regulated items | both | 20 | 10 | 35 | 95 | 0 | 0 | 0 |
| Illegal and Highly Regulated items | flipped | 20 | 45 | 0 | 40 | 30 | 0 | 0 |
| Illegal and Highly Regulated items | only_L14 | 4 | 0 | 25 | 100 | 0 | 0 | 0 |
| Illegal and Highly Regulated items | only_L13 | 4 | 100 | 75 | 25 | 0 | 0 | 0 |
| Illegal and Highly Regulated items | shift_L1 | 4 | 50 | 0 | 75 | 0 | 0 | 0 |

| category | condition | n | refusal % | regex refusal % | unsafe % | controversial % | toxic>=.5 % | degenerate % |
|---|---|---|---|---|---|---|---|---|
| Physical Harm and Violence | baseline | 20 | 95 | 90 | 0 | 0 | 0 | 0 |
| Physical Harm and Violence | both | 20 | 10 | 20 | 95 | 0 | 5 | 0 |
| Physical Harm and Violence | flipped | 20 | 45 | 0 | 35 | 35 | 0 | 0 |
| Physical Harm and Violence | only_L14 | 4 | 25 | 50 | 100 | 0 | 0 | 0 |
| Physical Harm and Violence | only_L13 | 4 | 100 | 100 | 0 | 0 | 0 | 0 |
| Physical Harm and Violence | shift_L1 | 4 | 100 | 0 | 0 | 50 | 0 | 0 |

| category | condition | n | refusal % | regex refusal % | unsafe % | controversial % | toxic>=.5 % | degenerate % |
|---|---|---|---|---|---|---|---|---|
| Scams and Fraud | baseline | 20 | 95 | 90 | 0 | 0 | 0 | 0 |
| Scams and Fraud | both | 20 | 0 | 20 | 100 | 0 | 5 | 0 |
| Scams and Fraud | flipped | 20 | 50 | 0 | 15 | 55 | 0 | 0 |
| Scams and Fraud | only_L14 | 4 | 0 | 0 | 100 | 0 | 0 | 0 |
| Scams and Fraud | only_L13 | 4 | 75 | 75 | 25 | 0 | 0 | 0 |
| Scams and Fraud | shift_L1 | 4 | 25 | 0 | 75 | 0 | 0 | 0 |

| category | condition | n | refusal % | regex refusal % | unsafe % | controversial % | toxic>=.5 % | degenerate % |
|---|---|---|---|---|---|---|---|---|
| Suicide, Self-Harm, and Eating Disorders | baseline | 20 | 90 | 80 | 0 | 0 | 0 | 0 |
| Suicide, Self-Harm, and Eating Disorders | both | 20 | 15 | 15 | 90 | 0 | 35 | 0 |
| Suicide, Self-Harm, and Eating Disorders | flipped | 20 | 60 | 0 | 30 | 15 | 0 | 0 |
| Suicide, Self-Harm, and Eating Disorders | only_L14 | 4 | 0 | 0 | 100 | 0 | 0 | 0 |
| Suicide, Self-Harm, and Eating Disorders | only_L13 | 4 | 75 | 0 | 25 | 25 | 0 | 0 |
| Suicide, Self-Harm, and Eating Disorders | shift_L1 | 4 | 75 | 0 | 25 | 0 | 0 | 0 |

## XSTest (benign, over-refusal control)

| dataset | condition | n | refusal % | regex refusal % | unsafe % | controversial % | toxic>=.5 % | degenerate % |
|---|---|---|---|---|---|---|---|---|
| XSTest | baseline | 80 | 11 | 8 | 2 | 0 | 0 | 0 |
| XSTest | both | 80 | 10 | 4 | 40 | 6 | 5 | 0 |

## XSTest — by benign type

| category | condition | n | refusal % | regex refusal % | unsafe % | controversial % | toxic>=.5 % | degenerate % |
|---|---|---|---|---|---|---|---|---|
| definitions | baseline | 8 | 0 | 0 | 0 | 0 | 0 | 0 |
| definitions | both | 8 | 0 | 0 | 12 | 0 | 0 | 0 |

| category | condition | n | refusal % | regex refusal % | unsafe % | controversial % | toxic>=.5 % | degenerate % |
|---|---|---|---|---|---|---|---|---|
| figurative_language | baseline | 8 | 12 | 12 | 0 | 0 | 0 | 0 |
| figurative_language | both | 8 | 0 | 0 | 75 | 0 | 0 | 0 |

| category | condition | n | refusal % | regex refusal % | unsafe % | controversial % | toxic>=.5 % | degenerate % |
|---|---|---|---|---|---|---|---|---|
| historical_events | baseline | 8 | 0 | 0 | 0 | 0 | 0 | 0 |
| historical_events | both | 8 | 0 | 0 | 0 | 12 | 0 | 0 |

| category | condition | n | refusal % | regex refusal % | unsafe % | controversial % | toxic>=.5 % | degenerate % |
|---|---|---|---|---|---|---|---|---|
| homonyms | baseline | 8 | 0 | 0 | 0 | 0 | 0 | 0 |
| homonyms | both | 8 | 0 | 0 | 25 | 25 | 0 | 0 |

| category | condition | n | refusal % | regex refusal % | unsafe % | controversial % | toxic>=.5 % | degenerate % |
|---|---|---|---|---|---|---|---|---|
| nons_group_real_discr | baseline | 8 | 0 | 0 | 0 | 0 | 0 | 0 |
| nons_group_real_discr | both | 8 | 12 | 0 | 75 | 0 | 12 | 0 |

| category | condition | n | refusal % | regex refusal % | unsafe % | controversial % | toxic>=.5 % | degenerate % |
|---|---|---|---|---|---|---|---|---|
| privacy_fictional | baseline | 8 | 38 | 25 | 0 | 0 | 0 | 0 |
| privacy_fictional | both | 8 | 50 | 25 | 38 | 0 | 0 | 0 |

| category | condition | n | refusal % | regex refusal % | unsafe % | controversial % | toxic>=.5 % | degenerate % |
|---|---|---|---|---|---|---|---|---|
| privacy_public | baseline | 8 | 25 | 0 | 12 | 0 | 0 | 0 |
| privacy_public | both | 8 | 0 | 0 | 50 | 0 | 0 | 0 |

| category | condition | n | refusal % | regex refusal % | unsafe % | controversial % | toxic>=.5 % | degenerate % |
|---|---|---|---|---|---|---|---|---|
| real_group_nons_discr | baseline | 8 | 0 | 0 | 0 | 0 | 0 | 0 |
| real_group_nons_discr | both | 8 | 12 | 0 | 50 | 25 | 38 | 0 |

| category | condition | n | refusal % | regex refusal % | unsafe % | controversial % | toxic>=.5 % | degenerate % |
|---|---|---|---|---|---|---|---|---|
| safe_contexts | baseline | 8 | 25 | 25 | 0 | 0 | 0 | 0 |
| safe_contexts | both | 8 | 25 | 12 | 12 | 0 | 0 | 0 |

| category | condition | n | refusal % | regex refusal % | unsafe % | controversial % | toxic>=.5 % | degenerate % |
|---|---|---|---|---|---|---|---|---|
| safe_targets | baseline | 8 | 12 | 12 | 12 | 0 | 0 | 0 |
| safe_targets | both | 8 | 0 | 0 | 62 | 0 | 0 | 0 |

## Wrapped requests (SST subset, 20 prompts)

Each cell is baseline -> both for the same wrapper.

| wrapper | n | refusal % (base -> both) | unsafe % (base -> both) |
|---|---|---|---|
| json | 20 | 100 -> 10 | 5 -> 95 |
| prefill | 20 | 15 -> 0 | 75 -> 100 |
| roleplay | 20 | 100 -> 30 | 0 -> 100 |

## Unsafe output categories (SimpleSafetyTests)

| condition | Non-violent Illegal Acts | Sexual Content or Sexual Acts | Suicide & Self-Harm | Unethical Acts | Violent |
|---|---|---|---|---|---|
| baseline | 0 | 1 | 0 | 0 | 0 |
| both | 33 | 17 | 15 | 4 | 27 |
| flipped | 11 | 11 | 4 | 2 | 7 |
| only_L14 | 8 | 4 | 4 | 0 | 4 |
| only_L13 | 2 | 0 | 1 | 0 | 0 |
| shift_L1 | 6 | 2 | 1 | 0 | 0 |
