# Step 2.5 — Team + Player Profile Crowdsource Results
# Crowdsourced: 2026-03-19 (Session 209)
# Models: GPT-4.1, Grok-4-1-fast-reasoning, Kimi-K2.5
# Status: RAW — needs Peter review to lock final criteria


## TEAM PROFILES — GPT-4.1

1. Recent Win Percentage | 9 | Last 5 games results | Proportion of games won in the last 5 games, reflecting current form.  
2. Season Win Percentage | 7 | Season win-loss record | Overall season success rate, indicating baseline team strength.  
3. Offensive Rating (Recent) | 8 | Offensive rating per 100 possessions (last 5 games) | Teamâ€™s points scored per 100 possessions over the last 5 games, capturing short-term scoring efficiency.  
4. Defensive Rating (Recent) | 9 | Defensive rating per 100 possessions (last 5 games) | Points allowed per 100 possessions over the last 5 games, reflecting current defensive strength.  
5. Net Rating (Recent) | 10 | Difference between offensive and defensive rating (last 5 games) | Overall team performance margin in the last 5 games.  
6. Effective Field Goal % (Recent) | 7 | eFG% last 5 games | Shooting efficiency adjusted for 3-pointers in recent games.  
7. Turnover Percentage (Recent) | 6 | Turnovers per 100 possessions (last 5 games) | How often the team turns the ball over in recent games.  
8. Assist-to-Turnover Ratio (Recent) | 6 | Assists/Turnovers (last 5 games) | Ball movement and decision-making efficiency in recent games.  
9. Opponent 3PT% Allowed (Recent) | 7 | Opponent 3PT% (last 5 games) | How well the team defends the 3-point line lately.  
10. Rebound Differential (Recent) | 6 | Team vs. opponent total rebounds (last 5 games) | Edge or deficit on the boards in recent games.  
11. Injury Impact Score | 9 | Injury report, minutes lost by key players (last 5 games) | Weighted measure of key players unavailable or playing limited minutes.  
12. Back-to-Back Status | 7 | Schedule (NBA calendar) | Whether the team is playing on consecutive nights, which can affect fatigue.  
13. Travel Distance (Last 5 Games) | 5 | Team travel logs, game locations | Total miles traveled over last 5 games, indicating potential fatigue.  
14. Pace (Recent) | 5 | Possessions per 48 min (last 5 games) | How fast the team has been playing recently.  
15. Clutch Performance | 6 | +/- in games within 5 points in last 5 minutes (season & last 5 games) | Success in close games, both recent and overall.  
16. Bench Scoring Contribution (Recent) | 5 | Bench points per game (last 5 games) | Points provided by non-starters, indicating depth.  
17. Opponent Strength (Recent) | 8 | Opponent win % in last 5 games | Average quality of teams faced recently.  
18. Free Throw Rate (Recent) | 4 | Free throws per field goal attempt (last 5 games) | Ability to generate and convert free throws in recent games.  
19. Turnover Forced Percentage (Recent) | 5 | Opponent turnovers per 100 possessions (last 5 games) | Defensive disruptiveness in recent play.  
20. Home/Away Differential (Recent & Season) | 6 | Home vs. away win % (last 5 games, season) | Performance gap between home and road games, which can affect upcoming matchups.

## TEAM PROFILES — GROK

1. Recent Form | 10 | Win-loss record and point differential over last 5 games vs season averages | Captures the team's momentum and short-term consistency in results.

2. Net Rating | 9 | Net Rating (OffRtg - DefRtg) averaged from last 5 games and season | Provides a pace-adjusted measure of overall team dominance.

3. Offensive Efficiency | 8 | Offensive Rating (points/100 possessions) from last 5 games and season avg | Evaluates scoring productivity independent of pace.

4. Defensive Efficiency | 9 | Defensive Rating (points allowed/100 possessions) from last 5 games and season avg | Assesses ability to limit opponent scoring.

5. Effective FG% | 7 | Team eFG% from last 5 games and season avg | Measures shooting efficiency weighting threes higher.

6. Turnover Rate | 8 | Turnover % (TO/possessions) from last 5 games and season avg | Gauges ball-handling reliability under pressure.

7. Offensive Rebound % | 7 | ORB% from last 5 games and season avg | Indicates control of missed shots for second chances.

8. Free Throw Rate | 6 | FTR (FTA/FGA) from last 5 games and season avg | Reflects aggressiveness in drawing fouls.

9. Pace Factor | 5 | Possessions per 48 minutes from last 5 games and season avg | Influences matchup tempo and fatigue.

10. 3PT Dependency | 6 | 3PA/FGA ratio from last 5 games and season avg | Highlights reliance on perimeter shooting volume.

11. Home/Away Split | 7 | Net Rating and win % split by home/away from last 5 games and season | Accounts for location-based performance edges.

12. Rest Days Avg | 8 | Team rest days vs opponent rest days averaged over last 5 games | Factors schedule fatigue differential.

13. B2B Performance | 7 | Net Rating on back-to-back games from last 5 relevant games and season | Identifies short-rest vulnerabilities.

14. Injury Impact | 9 | Aggregate Win Shares or BPM of missing key players (>20% minutes) from last 5 affected games | Quantifies roster depth losses.

15. Clutch Rating | 8 | Net Rating in clutch time (last 5 min, score +/-5 pts) from last 5 games and season | Tests closing ability in tight games.

16. Bench Net Rating | 6 | Net Rating with bench-heavy lineups (>50% bench minutes) from last 5 games and season | Measures reserve unit effectiveness.

17. Opponent Strength | 7 | Avg opponent Net Rating faced over last 5 games | Adjusts recent form for competition quality.

18. Transition Efficiency | 5 | Fast break points per game from last 5 games and season avg | Evaluates open-court scoring prowess.

## TEAM PROFILES — KIMI K2.5

1. **Offensive Rating Momentum** | 9 | Last 5 games OFF_RTM vs season OFF_RTM average | Measures recent offensive efficiency trend against baseline performance.

2. **Defensive Rating Momentum** | 10 | Last 5 games DEF_RTM vs season DEF_RTM average | Quantifies defensive effectiveness trend and suppression capability.

3. **Rest Equity Index** | 8 | Days of rest (0/1/2/3+) differential vs opponent | Calculates scheduling fatigue advantage or disadvantage.

4. **Injury Adjusted NET** | 9 | Weighted on/off court plus-minus of injured/returning players Ã— projected minutes | Estimates lineup impact from roster availability changes.

5. **True Shooting Volatility** | 5 | Last 5 games TS% standard deviation vs season-long variance | Captures scoring consistency risk and offensive variance.

6. **Perimeter Defense Regression** | 7 | Last 5 opponent 3P% vs opponent season 3P% average | Identifies if recent 3-point defense is skill-based or variance-driven luck.

7. **Pace Control Delta** | 6 | Last 5 possessions per game vs opponent season pace differential | Measures ability to impose preferred tempo or adapt to opponent style.

8. **Back-to-Back Burden** | 7 | Binary B2B flag + cumulative air miles traveled in last 7 days | Flags acute fatigue from consecutive games plus travel load.

9. **Bench Unit Efficiency** | 6 | Non-starter NET_RATING in last 5 games | Evaluates second unit performance and depth advantage during recent stretch.

10. **Clutch Execution Score** | 7 | Last 5 games: eFG% + FT% - TOV% in last 5 minutes within 5 points | Grades late-game decision making and pressure situational performance.

11. **Transition Defense Suppression** | 7 | Last 5 games: opponent fast break points per 100 possessions allowed | Measures ability to prevent easy transition baskets and floor balance.

12. **Free Throw Generation Trend** | 5 | Last 5 FTA per 100 possessions vs season average | Tracks recent rim pressure and foul-drawing aggression.

13. **Rebounding Dominance** | 6 | Last 5 games OREB% minus opponent OREB% differential | Quantifies possession control via offensive and defensive glass.

14. **Strength of Schedule Adjustment** | 8 | Last 5 opponents' average NET_RATING vs upcoming opponent NET_RATING | Contextualizes recent performance against level of competition faced.

15. **Assist-to-Turnover Momentum** | 6 | Last 5 AST/TO ratio vs season AST/TO ratio | Indicates ball security quality and offensive system execution flow.

16. **Paint Protection Grade** | 7 | Last 5 opponent restricted area FG% vs league average plus block rate | Measures interior defensive deterrence and rim protection.

17. **Circadian Disruption Index** | 5 | Binary indicators for 2+ time zone crossings or altitude games (DEN/UTA) | Accounts for physiological travel penalties and sleep cycle disruption.

18. **Defensive Turnover Creation** | 6 | Last 5 opponent TOV% vs league average TOV% | Measures disruptive defense and live-ball turnover generation for easy points.

## PLAYER PROFILES — GPT-4.1

1. usage_rate | 9 | NBA advanced stats (USG%) | Measures the percentage of team plays used by the player while on the floor, indicating offensive involvement.

2. recent_scoring_trend | 8 | Last 5-10 game logs (PTS) | Tracks changes in the player's scoring output over recent games to assess form.

3. assist_opportunities | 7 | Player tracking (NBA.com, Second Spectrum) | Number of potential assists per game, capturing playmaking opportunities beyond raw assists.

4. rebound_chances | 6 | Player tracking (contested & uncontested rebounds) | Frequency of rebounding opportunities, not just total rebounds, to gauge involvement on the boards.

5. defensive_matchup_difficulty | 8 | Opponent defensive ratings, DRTG vs. position | Estimates the defensive challenge faced based on expected primary defenders and team schemes.

6. minutes_projection | 10 | Coach quotes, past games, rotation trends | Expected playing time, which is crucial for all prop and impact analyses.

7. injury_risk | 9 | Injury reports, historical DNPs, back-to-backs | Likelihood of limited minutes or performance drop due to injuries or rest.

8. foul_trouble_risk | 5 | Personal fouls per 36, opponent draw foul rate | Probability of reduced minutes due to fouls, especially against certain matchups.

9. matchup_advantage | 8 | Opponent positional weaknesses, on/off splits | Quantifies how favorable the matchup is relative to league average at the playerâ€™s position.

10. usage_on_back_to_back | 6 | Historical USG% in back-to-backs, schedule | Measures how usage changes on consecutive game days for fatigue impact.

11. pace_adjustment_factor | 5 | Team pace, opponent pace | Adjusts projections based on expected possessions in the upcoming game.

12. shooting_efficiency | 7 | True shooting % (TS%), eFG%, shot location data | Captures scoring efficiency and shot selection quality.

13. volatility_index | 6 | Standard deviation of last 10 game stat lines | Assesses prop risk by measuring consistency in recent performances.

14. on_off_impact | 6 | Team net rating with player on/off court | Evaluates the player's overall impact on team performance.

15. clutch_performance | 4 | NBA clutch stats (last 5 min, +/- 5 pts) | Measures ability and likelihood to perform in close games, affecting props and team outcome.

16. assist_turnover_ratio | 5 | AST/TO from game logs | Assesses ball security and playmaking reliability.

17. role_change_flag | 7 | Recent lineup changes, depth chart, coach comments | Detects any recent or expected change in role (starter/bench, usage spike/drop).

18. free_throw_rate | 4 | FTA/FGA, opponent foul propensity | Indicates how often the player gets to the line, boosting scoring prop reliability.

19. 3pt_volume_and_accuracy | 5 | 3PA, 3P% game logs, opponent 3pt defense | Evaluates three-point shooting threat and matchup influence.

20. hustle_stats | 3 | Contested shots, deflections, loose balls (NBA tracking) | Captures non-box score efforts that signal engagement and possible stat boosts.

## PLAYER PROFILES — GROK

1. Usage Rate | 9 | Last 10 games USG%, season USG% | Measures the percentage of team plays ending in a player usage event, predicting volume for points, assists, and rebounds props.  
2. Minutes Projection | 10 | Season avg MPG, last 5 games MPG, coach rotation data | Forecasts playing time that scales all counting stats and team impact.  
3. Injury Risk Score | 10 | Games missed last 30 days, load management %, injury history | Gauges probability of DNP, limited minutes, or reduced performance due to health.  
4. Recent PPG Form | 8 | Avg points last 5 games vs season avg | Captures scoring hot/cold streaks for over/under points props.  
5. Rebound % | 7 | Last 10 games REB%, opponent REB% allowed | Player's share of available rebounds, key for REB props and possession battles.  
6. Assist % | 7 | Last 10 games AST%, team assist rate | Percentage of teammate field goals assisted, driving AST props and offensive flow.  
7. True Shooting Efficiency | 6 | Last 10 games TS%, matchup vs opponent DRTG | Overall scoring efficiency accounting for shot types, vital for PRA and team offense.  
8. 3PT Volume | 7 | Last 10 games 3PA, opponent 3P% allowed | Three-point attempt rate against defensive schemes, central to 3PM props.  
9. Free Throw Rate | 6 | Last 10 games FTA per 36 min, opponent foul rate | Frequency of drawing fouls, boosting points and reliability in close games.  
10. Turnover % | 5 | Last 10 games TOV%, high USG% games | Turnover frequency relative to usage, hurting props and team possessions.  
11. Pace Adjusted Fantasy Points | 9 | Last 10 games FanDuel points per 36 min, adjusted for pace | Standardized production metric predicting prop overs and lineup impact.  
12. Matchup Defensive Exploit | 8 | Player's wOBA vs opponent DBPM, specific positional D | Advantage against weak defenders or schemes for elevated usage and efficiency.  
13. Defensive Rating | 5 | Last 10 games DRtg, steals/blocks per 36 | Points allowed per 100 possessions, influencing team win probability and minutes.  
14. Back-to-Back Fatigue | 7 | Performance delta on 0/1/2+ rest days last 30 games | Quantifies drop-off in stats on no rest, critical for B2B props.  
15. Recent Prop Hit Rate | 8 | Beat prop line % last 10 games (PTS, REB, AST) | Momentum indicator for beating betting lines based on form.  
16. Opponent Pace Rank | 6 | Opponent season pace percentile | Faster opponent tempo increases possessions and counting stat opportunities.  
17. Home/Away Split | 5 | Stat differentials home vs away last 20 games | Performance variance by venue, affecting props and travel fatigue.  
18. Team Implied Total | 9 | Vegas implied team points for game | Scales individual player output based on expected team scoring volume.  
19. Steal/Block Combo | 4 | Last 10 games STL% + BLK%, opponent TO% forced | Defensive disruption potential for niche props and transition scoring.  
20. Rest Days Efficiency | 6 | TS% and USG% delta after 2+ rest vs 0 rest | Non-obvious recovery boost post-rest, enhancing next-game impact.

## PLAYER PROFILES — KIMI K2.5

1. **Usage Velocity** | 10 | USG%, Minutes per game, Time of Possession, Frontcourt touches | Measures current offensive involvement rate relative to season baseline to predict counting stat volume.

2. **Matchup Defensive Rating** | 9 | D-LEBRON, Matchup difficulty data, Opponent positional defensive efficiency allowed | Quantifies the defensive quality of primary matchup assignment and scheme vulnerability to playerâ€™s skill set.

3. **Minutes Security Index** | 9 | Rotation patterns, Coach substitution tendencies, Foul rate, Blowout probability | Predicts floor time stability excluding unpredictable variables like disqualification or garbage time variance.

4. **Recent Form Trajectory** | 9 | Last 5-10 game rolling averages, Z-scores vs season means, True Shooting variance | Captures momentum direction and statistical deviation from baseline performance trends.

5. **Injury/Load Management Risk** | 8 | Injury reports, B2B status, Minutes trend (last 3 games), Load management history | Binary risk assessment for sudden minute restrictions, absence probability, or burst limitation.

6. **Pace Delta Impact** | 7 | Team pace vs Opponent pace, Possessions per 48, Transition frequency | Calculates possession volume opportunity relative to playerâ€™s typical game environment and stamina profile.

7. **Assist Opportunity Matrix** | 7 | Potential assists, Teammate 3PT%, Drive-and-kick frequency, Hockey assists | Measures quality of passing opportunities independent of teammate conversion variance.

8. **Rebounding Opportunity Share** | 7 | Offensive/defensive rebound chances, Box-out rate, Opponent projected miss rate | Calculates available rebound volume based on matchup shooting profiles and positioning data.

9. **Free Throw Generation Rate** | 7 | FTr (Free Throw Rate), Drive frequency, Referee crew foul rates, Paint touches | Predicts easy scoring volume and foul drawing independent of jump shooting variance.

10. **Lineup Context Volatility** | 7 | Teammate availability, Lineup continuity score, Positionless switches, Backup PG status | Impact of roster changes on role definition, usage redistribution, and statistical opportunity.

11. **Defensive Assignment Difficulty** | 7 | Minutes spent guarding high-usage opponents, Defensive load, Ball-screen coverage frequency | Energy expenditure metric affecting offensive efficiency on opposite end due to defensive fatigue.

12. **Shooting Regression Indicator** | 6 | eFG%, ShotQuality/Second Spectrum data, Luck variance (actual vs expected FG%) | Identifies if recent shooting splits are sustainable or due for mean reversion correction.

13. **Rest & Recovery Status** | 6 | Days rest, Travel miles (circadian disruption), Altitude change, Back-to-back index | Physiological readiness indicator affecting burst capacity, shooting legs, and cognitive decision speed.

14. **Clutch Usage Premium** | 6 | 4th quarter usage rate, Close game minutes probability, Go-to scorer status | Likelihood of high-leverage minutes where usage concentrates on primary options late in games.

15. **Home/Road Splits Variance** | 6 | Home vs Away statistical differentials, Sleep score, Familiar rim/background effects | Environmental performance gaps affecting efficiency and volume in non-neutral venues.

16. **Foul Trouble Probability** | 5 | Personal fouls per 36, Defensive aggression score, Referee crew whistle tendencies | Risk assessment for minute disruption due to disqualification or forced defensive conservatism.

17. **Blowout Variance Factor** | 5 | Point spread, Team rest differential, Garbage time probability, Bench unit quality | Risk of reduced minutes in decided games or unexpected extended run if reserve unit struggles.

18. **Historical Head-to-Head** | 5 | Career vs opponent stats, Specific defender history, Arena familiarity, Coaching scheme memory | Sample-specific tendencies against current opponent personnel and tactical adjustments.
