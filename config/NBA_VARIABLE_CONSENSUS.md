# NBA Variable Consensus Matrix
# Crowdsourced from 6 AI models: GPT-4.1, Grok, GPT-5.1, DeepSeek, o4-mini, Kimi K2.5
# Date: 2026-03-19

VARIABLE                             MODELS  AVG WT    RANGE  DESCRIPTION
--------------------------------------------------------------------------------------------------------------
defensive efficiency                      5     9.8     9-10  Variance between season-long defensive rating and performance over the last 8-10
offensive efficiency                      5     9.6     8-10  Points per possession generated in non-transition sets, isolating execution agai
injury/star impact                        5     9.0     7-10  Impact rating of injuries, load management, or minute restrictions on rotation d
rest/fatigue                              5     7.8     5-10  Specific sequencing fatigue including back-to-backs, altitude transitions, or fo
sharp money/line movement                 5     7.3      3-9  Degree to which the betting line moves against public money percentages, signali
three-point matchup                       5     6.8      5-8  Differential between recent opponent three-point shooting luck and expected regr
rebounding                                5     6.6      6-7  Team offensive rebounding percentage relative to opponent's defensive box-out ef
turnover differential                     5     6.5      4-8  Variance between turnover count and actual points allowed off those turnovers, i
pace/tempo                                5     6.4      6-7  Degree of tempo clash between teams' average possession length and opponent's ab
bench depth                               5     6.0      6-6  Variance in second-unit offensive rating plus net rating during non-overlapping 
matchup exploit                           4     6.8      5-8  Quantified switchability index measuring ability to defend multiple actions with
clutch performance                        4     6.2      6-7  Difference between actual win percentage and expected win percentage in games de
free throw                                4     5.1      4-6  Differential in free throw attempts per field goal attempt between teams, indica
coaching                                  3     7.0      6-8  Second-half ATS performance and timeout effectiveness metrics, measuring in-game
shooting variance/regression              3     6.7      6-8  Opponent shot distribution weighted by expected value, measuring defensive schem
travel/altitude                           3     6.7      6-7  Total miles traveled by each team in prior 72 hours, with cross-timezone penalty
motivation/letdown                        3     6.7      6-7  Situational psychology metric measuring distraction potential from upcoming marq
ATS trend                                 3     6.2      4-8  Evaluates each teamâ€™s recent success covering spreads, reflecting market mispr
rim/paint                                 2     8.0      8-8  Mismatch between team's rim attempt rate and opponent's block percentage plus re
home/away                                 2     7.5      7-8  Measures the difference in team performance at home versus away, adjusting for v
net rating                                2     7.5      6-9  Team's point differential per 100 possessions in "clutch" minutes (last 5 mins, 
referee                                   2     6.0      5-7  Specific officiating crew's historical impact on game total, free throw rate, an
public betting                            2     5.0      4-6  Degree to which betting action is lopsided on one side, indicating potential lin
other: Recent Form                        1     8.0      8-8  Scores performance over the last 5 games, capturing momentum and current level.
other: Recent Form (Last 5 Games)         1     8.0      8-8  Weighted performance over the last five games, adjusting for opponent strength.
other: Pick-and-Roll Defense              1     7.0      7-7  Opponent's points per possession allowed on PnR vs team's PnR efficiency.
other: Starting Five Synergy              1     7.0      7-7  On-court net rating of projected starting lineup vs opponent's.
transition                                1     7.0      7-7  Differential between team's transition points scored per possession and opponent
other: Defensive_Connectivity             1     7.0      7-7  Help-defense rotation speed and communication effectiveness quantified through p
head-to-head                              1     6.0      6-6  Scores how teams have performed against each other recently, indicating matchup 
other: Homeâ€‘Court Environment Strength       1     6.0      6-6  Uses historical ATS performance, altitude, crowd intensity, and travel difficult
lineup continuity                         1     6.0      6-6  Measures games played with the same starting lineup or core rotation and its net
other: Defensive Scheme Adjustment        1     6.0      6-6  Ability of a team's defense to specifically counter the opponent's primary offen
other: Revenge Factor                     1     5.0      5-5  Team's performance in a rematch against a team that recently defeated them.
other: Offensive Style Clash              1     5.0      5-5  How well one team's offensive scheme attacks the other's specific defensive weak
other: Post-All-Star Break Trend          1     5.0      5-5  Team's performance shift (offensive/defensive rating) since the All-Star break.
other: Time Zone Change Impact            1     4.0      4-4  Road team's performance when crossing two or more time zones for a game.


## RECOMMENDED TOP 25 (sorted by consensus strength)

| # | Variable | Models | Avg Weight | Recommendation |
|---|----------|--------|------------|----------------|
| 1 | defensive efficiency | 5/6 | 9.8 | LOCK |
| 2 | offensive efficiency | 5/6 | 9.6 | LOCK |
| 3 | injury/star impact | 5/6 | 9.0 | LOCK |
| 4 | rest/fatigue | 5/6 | 7.8 | LOCK |
| 5 | sharp money/line movement | 5/6 | 7.3 | LOCK |
| 6 | three-point matchup | 5/6 | 6.8 | LOCK |
| 7 | rebounding | 5/6 | 6.6 | LOCK |
| 8 | turnover differential | 5/6 | 6.5 | LOCK |
| 9 | pace/tempo | 5/6 | 6.4 | LOCK |
| 10 | bench depth | 5/6 | 6.0 | LOCK |
| 11 | matchup exploit | 4/6 | 6.8 | LOCK |
| 12 | clutch performance | 4/6 | 6.2 | LOCK |
| 13 | free throw | 4/6 | 5.1 | LOCK |
| 14 | coaching | 3/6 | 7.0 | STRONG |
| 15 | shooting variance/regression | 3/6 | 6.7 | STRONG |
| 16 | travel/altitude | 3/6 | 6.7 | STRONG |
| 17 | motivation/letdown | 3/6 | 6.7 | STRONG |
| 18 | ATS trend | 3/6 | 6.2 | STRONG |
| 19 | rim/paint | 2/6 | 8.0 | REVIEW |
| 20 | home/away | 2/6 | 7.5 | REVIEW |
| 21 | net rating | 2/6 | 7.5 | REVIEW |
| 22 | referee | 2/6 | 6.0 | REVIEW |
| 23 | public betting | 2/6 | 5.0 | REVIEW |
| 24 | other: Recent Form | 1/6 | 8.0 | REVIEW |
| 25 | other: Recent Form (Last 5 Games) | 1/6 | 8.0 | REVIEW |
