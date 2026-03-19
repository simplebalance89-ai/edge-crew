# All Sports — Team + Player Profile Crowdsource Results
# Source: Kimi K2.5 + DeepSeek V3 via Azure AI Services
# Date: 2026-03-19 (Session 209)
# Status: RAW — needs Peter review per sport


============================================================
## NBA
============================================================

### TEAM PROFILE

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

### PLAYER PROFILE

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

============================================================
## NCAAB
============================================================

### TEAM PROFILE

Adj. Defensive Efficiency Grade | 10 | BartTorvik | Blend of season-long adjusted defensive efficiency with last 5 games trend weighted 70/30 to isolate recent form from schedule noise.

Adj. Offensive Efficiency Grade | 10 | BartTorvik | Composite score blending season adjusted offensive efficiency with last 5 games effective field goal percentage and turnover rates to capture current offensive health.

Half-Court Grind Efficiency | 8 | Synergy | Points per possession scored in non-transition sets over last 5 games versus season average, filtering out fast-break noise that casual bettors overweight.

True Road Resilience | 8 | TeamRankings | Differential between away-game adjusted efficiency margins and home performance over last 5 road contests versus season baseline to quantify genuine travel immunity.

Clutch Execution Rating | 8 | SportsReference | Offensive efficiency in last 5 minutes of single-possession games over last 5 contests compared to season crunch-time performance for late-game cover probability.

Pace Control Index | 8 | BartTorvik | Measures team's ability to impose its preferred tempo over last 5 games compared to season average, critical for totals betting market inefficiencies.

Live Ball Turnover Vulnerability | 7 | Synergy | Percentage of opponent turnovers forced that are steals (live ball) in last 5 games, weighted by season transition defense efficiency to predict easy transition points allowed.

Offensive Rebounding Dominance | 7 | BartTorvik | Offensive rebounding percentage in last 5 games adjusted for opponent defensive rebounding strength versus season-long conference-adjusted rate for hidden possession value.

Transition Defense Vulnerability | 7 | Synergy | Points per possession allowed in transition over last 5 games compared to season average, accounting for opponent transition frequency faced to expose athletic mismatches.

Rest & Travel Fatigue Index | 7 | NCAA Schedule Data | Calculated fatigue score combining miles traveled, time zones crossed, and days of rest preceding the game relative to season average recovery windows.

3-Point Variance Risk | 7 | NCAA Stats | Quantifies reliance on 3PT shooting volume in last 5 games versus season-long accuracy to flag high-volatility offenses prone to regression against the spread.

Def

### PLAYER PROFILE

**Usage Rate** | 10 | KenPom/Sports-Reference | Percentage of team possessions ending with player's shot, turnover, or free throw attempt while on floor.  
**Defensive Matchup Difficulty** | 9 | ShotQuality/CBB Analytics | Opponent's defensive efficiency allowed specifically to player's primary position and play type.  
**Blowout Minutes Risk** | 9 | Win Probability Models | Likelihood of game decided by 15+ points resulting in benching of starters during final 8 minutes.  
**Rebounding Opportunity Index** | 8 | Team/Opponent Box Scores | Combined team offensive rebound rate and opponent defensive rebound vulnerability creating rebound chances per minute.  
**Foul Trouble Probability** | 8 | Box Score Logs | Personal fouls committed per 40 minutes against opponents with high foul-drawing rates at player's position.  
**Pace Delta** | 8 | KenPom Adjusted Tempo | Projected possession differential between player team tempo and opponent pace allowed affecting counting stat volume.  
**Free Throw Generation Rate** | 7 | Play-by-Play Data | Free throw attempts per 100 possessions independent of shooting variance providing points floor.  
**Conference Play Adjustment** | 7 | Conference-Only Splits | Efficiency differential between non-conference and conference performance indicating true talent level against comparable athletes.  
**Back-to-Back Fatigue Factor** | 7 | Schedule API | Performance degradation in second game of tournament weekends or consecutive-day series common in NCAAB.  
**True Shooting Variance** | 6 | Last 5 Game Logs | Standard deviation of shooting efficiency indicating prop over/under volatility risk.  
**Three-Point Dependency** | 6 | Play-by-Play | Percentage of total points derived from three-pointers indicating high-variance scoring profile.  
**Clutch Usage Rate** | 6 | Sports-Reference Clutch | Percentage of team shots taken in last 5 minutes of single-digit games affecting late-game scoring upside.  
**Assist Network Stability** | 6 | Lineup Data | Percentage of team's assists generated by player versus system ball-movement indicating playmaking volatility.  
**Rest Status** | 6 | Schedule API | Days of rest (0/1/2+) since last contest affecting shooting legs and recovery metrics.  
**Defensive Stat Opportunity** | 5 | Opponent Stats | Opponent's turnover percentage and block susceptibility rate projecting steals and blocks prop ceiling.  
**Home/Away Efficiency Delta** | 5 | Splits Data | Variance between home and away performance metrics for location-based prop line adjustments.  
**Transition Frequency** | 5 | Synergy Sports | Percentage of offensive possessions in transition versus half-court affecting easy scoring opportunities.

============================================================
## NHL
============================================================

### TEAM PROFILE

5v5 Score-Adjusted Expected Goals For % | 10 | Natural Stat Trick | Measures puck possession and shot quality differential at even strength over the last 5 games versus season baseline to isolate sustainable offensive zone dominance.  
High-Danger Conversion Variance (Last 5) | 7 | MoneyPuck | Compares actual goals scored from high-danger areas to expected goals in last 5 games to flag unsustainable finishing luck due for regression.  
PDO (Last 5 Games) | 6 | Natural Stat Trick | Sums shooting percentage and save percentage from last 5 games to detect teams running hot or cold relative to the league mean of 100.  
Goaltender Goals Saved Above Expected (Last 5) | 9 | Evolving Hockey | Quantifies how many goals the starting goalie prevented relative to shot quality faced in last 5 outings to isolate true form from defensive help.  
Backup Goaltender Reliability Score | 8 | Daily Faceoff + Hockey Reference | Evaluates save percentage and high-danger save percentage of projected backup goalie in spot starts, weighted by career sample size.  
Rest Advantage vs Opponent | 8 | NHL Schedule API | Calculates the difference in days of rest between teams, penalizing teams on 0-1 days rest facing opponents on 3+ days.  
Special Teams Efficiency Differential | 7 | NHL API | Subtracts penalty kill percentage from power play percentage but adjusts for recent power play opportunities per game to account for referee assignment bias.  
Offensive Zone Start % (Last 5) | 6 | Natural Stat Trick | Tracks percentage of non-neutral zone starts in the offensive zone to identify teams receiving favorable deployment that inflates point production temporarily.  
Penalty Discipline Index | 7 | NHL API | Measures minor penalties taken per 60 minutes minus penalties drawn in last 5 games to quantify possession-killing aggression and referee targeting.  
Faceoff Win % in Offensive Zone (Last 5) | 6 | NHL API | Isolates faceoff percentage in the attacking zone to gauge immediate puck recovery and set play execution that sustains offensive pressure.  
High-Danger Save Percentage (Last 5) | 7 | Natural Stat Trick | Tracks save percentage on shots from the slot and crease area to assess scramble coverage and rebound control under pressure.  
3rd Period Score-Adjusted Corsi % (Last 5) | 7 | Natural Stat Trick | Measures shot attempt differential in the final frame when score effects are removed to identify teams with late-game territorial control and empty-net resistance.  
Expected Goals Against Per 60 (5v5) | 8 | MoneyPuck | Calculates expected goals allowed per 60 minutes at even strength across last 5 games to isolate defensive structure quality independent of goaltender performance.  
Roster Fatigue Index | 6 | NHL API + Travel Math | Aggregates games played in last 10 days, total travel miles, and time zone shifts to quantify cumulative physical load before matchup.  
Power Play Opportunities Per Game (Last 5) | 5 | NHL API | Tracks referee-assessed power play chances to detect teams benefiting from recent officiating tendencies that may not continue.  
Shooting Talent Regression Marker | 5 | Evolving Hockey | Compares team shooting percentage from high-danger areas in last 5 games to their 3-year rolling average to flag aberrant finishing skill displays.  
Blocked Shot Rate (5v5) | 5 | Natural Stat Trick | Measures shots blocked per unblocked shot attempt against to evaluate defensive commitment to shot suppression and shot-pass lanes.  
Goal Differential Momentum vs Season | 6 | NHL API | Compares goal differential in last 5 games to season average to distinguish legitimate form improvement from schedule-based variance.

### PLAYER PROFILE

**1. Momentum Index** | Weight: 9 | Data: Game logs (last 5 GP) | Description: Ratio of points-per-60 in last five games versus season average indicating short-term performance velocity relative to baseline.

**2. Deployment Stability** | Weight: 8 | Data: Shift charts, TOI logs | Description: Inverse coefficient of variation for total ice time over last five games reflecting coach trust and role consistency.

**3. Defensive Matchup Difficulty** | Weight: 9 | Data: Opponent projected pairings, historical line matching | Description: Composite score of opposing defense pair's expected goals against per 60 and likelihood of facing opponent's top checking line.

**4. Goaltender Vulnerability Alignment** | Weight: 8 | Data: Goalie save percentage by zone, player shot location heatmaps | Description: Correlation coefficient between player's high-danger shooting percentage and opposing goalie's high-danger save percentage weaknesses.

**5. Rest Differential** | Weight: 7 | Data: NHL Schedule API | Description: Difference in days of rest between player's team and opponent quantifying fatigue advantage or disadvantage.

**6. Venue Performance Delta** | Weight: 6 | Data: Home/away splits | Description: Normalized variance between home and away points

============================================================
## NFL
============================================================

### TEAM PROFILE

**Neutral Script Success Rate** | Weight: 10 | Data: nflfastR API | Description: Offensive success rate in neutral game states (win probability 20-80%) over last 5 games, filtering out garbage time noise that skews season scoring averages.

**Defensive Havoc Rate** | Weight: 9 | Data: SIS DataHub | Description: Percentage of defensive snaps generating TFL, pass breakup, or forced fumble, measuring disruptive ability independent of turnover luck or opponent quality.

**3rd Down Regression Index** | Weight: 8 | Data: nflfastR API | Description: Variance between last 5 games' 3rd down conversion rate and season baseline, flagging teams riding unsustainable streaks or suffering from temporary schematic fixes.

**Red Zone Finishing Efficiency** | Weight: 9 | Data: TruMedia/NFL GSIS | Description: Touchdown rate inside opponent 20-yard line weighted by goal-to-go frequency, distinguishing between field-goal merchants and spread-covering offenses.

**Pressure-to-Sack Conversion** | Weight: 7 | Data: PFF API | Description: Ratio of sacks to pressures generated over last 5 games, revealing secondary coverage quality and quarterback pocket presence under duress.

**Explosive Play Differential** | Weight: 8 | Data: nflfastR API | Description: Net difference between offensive plays gaining 20+ yards and defensive plays allowed per game, predictive of margin volatility independent of red zone variance.

**Pre-Snap Penalty Rate** | Weight: 6 | Data: NFL GSIS | Description: False starts and defensive offsides per 100 snaps over last 5 games, indicating offensive line communication breakdowns or defensive aggressive tendencies exploitable via hard count.

**Turnover Luck Factor** | Weight: 10 | Data: Football Outsiders / nflfastR | Description: Fumble recovery rate and interception rate compared to expected models based on ball placement, flagging immediate regression candidates for betting value.

**Pace Control Index** | Weight: 7 | Data: nflfastR (play timestamps) | Description: Average seconds per play in neutral game states, measuring tempo manipulation ability that impacts game totals and late comeback probability.

**Circadian Disadvantage Score** | Weight: 6 | Data: Schedule API / Time zone data | Description: Performance delta for Pacific Time teams playing 10am PT kickoffs versus standard 1pm PT starts, capturing biological rhythm disadvantages ignored by market openers.

**Short Rest Efficiency Delta** | Weight: 7 | Data: PFF API / Schedule data | Description: EPA per play differential between Thursday games or short-week road games versus standard rest, isolating coaching staff preparation efficiency and recovery protocols.

**Weather-Adjusted Pass Efficiency** | Weight: 8 | Data: OpenWeather API + nflfastR | Description: Quarterback ANY/A in games with wind speeds >12mph or precipitation versus dome conditions, essential for outdoor stadium totals and player prop adjustments.

**Offensive Line Continuity** | Weight: 7 | Data: Pro Football Reference injury database | Description: Percentage of offensive snaps over last 5 games played by season-long starting five, weighted by position importance (LT/C premium), predicting protection volatility.

**Blitz Vulnerability Index** | Weight: 6 | Data: SIS DataHub | Description: Offensive success rate against 6+ rushers over last 5 games, exposing immobile quarterbacks or slow-developing route trees susceptible to aggressive defensive gameplans.

**Special Teams Field Position** | Weight: 7 | Data: NFL GSIS | Description: Average starting field position after kickoffs and punts (excluding turnovers), representing hidden yards that compress or expand game totals independent of offense/defense efficiency.

**Garbage Time Noise Filter** | Weight: 8 | Data: nflfastR (WP model) | Description: Percentage of season scoring accrued while trailing by 14+ points in 4th quarter, subtracted from raw efficiency metrics to reveal true competitive performance levels.

**4th Down Aggression Success** | Weight: 6 | Data: nflfastR API | Description: Conversion rate on 4th downs with win probability between 15-85%, indicating coaching analytical alignment and critical late-game possession management.

**Closing Line Value Capture** | Weight: 9 | Data: Sportsbook API (Pinnacle/CRIS) | Description: Reverse line movement analysis comparing opening to closing numbers against public betting percentages, signaling sharp money positioning and market inefficiencies.

### PLAYER PROFILE

_Not retrieved — model lacks sufficient data for this sport/type_

============================================================
## MLB
============================================================

### TEAM PROFILE

**Bullpen Fatigue Index** | 9 | FanGraphs/PitchFX | Sum of high-leverage relief innings pitched over last 5 days weighted by appearances on consecutive days.

**SP Late-Game Endurance** | 8 | Baseball-Reference | Percentage of starts reaching the 6th inning or later over last 5 starts versus season baseline durability.

**Platoon Split Exploitation** | 7 | Statcast | wOBA differential of projected lineup against opposing starter's handedness compared to season average.

**High-Leverage Offense** | 8 | FanGraphs | OPS in late/close situations (7th+ inning, margin â‰¤2 runs) over the last 5 games versus season clutch performance.

**Batted Ball Regression Gap** | 7 | Statcast/xwOBA | Variance between actual wOBA and expected wOBA (xwOBA) over the rolling 5-game window indicating luck correction pending.

**Catcher Framing Value** | 6 | Baseball Prospectus | Strikes stolen above average per game by the catching battery over the last 5 contests affecting walk rates.

**Circadian Rhythm Disruption** | 6 | Schedule/Time Zone Data | Games played across 2+ time zones within 72 hours without an off-day, quantifying jet lag impact on reaction times.

**Defensive Positioning Efficiency** | 7 | Statcast OAA | Outs Above Average shifted specifically against the batted ball tendencies (pull/oppo/GB%) of the opposing lineup.

**Available High-Leverage Arms** | 9 | Team Reports/PitchFX | Count of rested elite relievers (0 days rest, sub-3.00 xFIP) available for deployment in late innings.

**First-Pitch Strike Momentum** | 6 | PitchFX | Starting pitcher's first-pitch strike percentage over last 5 starts predicting early inning control and pitch count efficiency.

**Park Factor Alignment** | 6 | FanGraphs Park Factors | Correlation between team's batted ball profile (GB/FB ratio, pull%) and daily venue's run suppression characteristics.

**Lineup Volatility Score** | 5 | Daily Lineups | Percentage of regular starters (top 7 by season plate appearances) active versus recent resting patterns.

**Umpire Zone Compatibility** | 6 | Umpire Historical Data | Pitcher command profile (edge percentage) matched against home plate umpire's historical strike zone expansion/contraction tendencies.

**Base Running Pressure Index** | 5 | Baseball-Reference BsR | Extra bases taken percentage plus steal attempts per opportunity over last 5 games creating defensive pressure.

**Pitch Tunneling Quality** | 7 | PitchFX/Statcast | Release point consistency and pitch movement convergence creating late break differentiation inducing whiffs.

**Getaway Day Rest Tendency** | 5 | Schedule Analysis | Final game of series before travel day indicating propensity for benching starters to avoid injuries.

### PLAYER PROFILE

**1. Weighted Recent Form (wOBA/ERA)**  
| Weight: 9 | Data Source: Statcast (14-day rolling) | Description: Measures short-term momentum against season baseline to identify players outperforming or underperforming their market lines.  

**2. Platoon Split Differential**  
| Weight: 8 | Data Source: FanGraphs (Career/Season) | Description: Quantifies performance gap vs. opposite-handed pitchers or hitters to exploit lefty-righty matchup advantages in prop pricing.  

**3. Opposing Matchup Quality**  
| Weight: 9 | Data Source: Statcast (xFIP/wOBA allowed) | Description: Grades the specific opponent's underlying skill to determine difficulty adjustments for strikeout, hit, or earned run props.  

**4. Ballpark Factor (Handedness-Specific)**  
| Weight: 7 | Data Source: Baseball-Reference (3-year regressed) | Description: Applies park-specific run and home run factors adjusted for batter handedness to forecast total base and HR prop viability.  

**5. Batting Order Position**  
| Weight: 8 | Data Source: Lineup APIs (RotoGrinders/ESPN) | Description: Determines projected plate appearance volume and RBI/run opportunity based on lineup slot (1-9) for counting stat props.  

**6. Batted Ball Luck Regression (xWOBA vs. wOBA)**  
| Weight: 8 | Data Source: Statcast (Barrel%, HardHit%) | Description: Identifies variance gaps between expected and actual outcomes to signal imminent positive or negative regression on hit/HR props.  

**7. Pitch Velocity/Stuff Trend**  
| Weight: 8 | Data Source: Statcast (Pitch tracking) | Description: Tracks 5-start velocity averages and spin rate changes to detect undisclosed injuries or mechanical adjustments affecting K props.  

**8. Times Through Order Penalty**  
| Weight: 8 | Data Source: FanGraphs (Splits by PA) | Description: Measures performance decay when facing hitters a third time to project starter longevity for quality start and win props.  

**9. Weather/Wind Vector**  
| Weight: 7 | Data Source: Weather APIs (Visual Crossing) | Description: Calculates wind speed/direction and temperature effects on batted ball distance to adjust over/under HR and total base lines.  

**10. Umpire Strike Zone Tendency**  
| Weight: 6 | Data Source: Umpire Scorecards (Zone size) | Description: Quantifies individual umpireâ€™s called strike zone size to project walk and strikeout rate deviations from pitcher season norms.  

**11. Bullpen Exhaustion Behind Starter**  
| Weight: 7 | Data Source: FanGraphs (Reliever usage) | Description: Aggregates bullpen pitches thrown in previous 48 hours to predict manager hook length and starter win probability.  

**12. Base Running Opportunity Index**  
| Weight: 5 | Data Source: Statcast (Pitcher slide-step time, Catcher pop time) | Description: Scores stolen base probability using pitcher delivery speed and catcher arm ratings for SB attempt props.  

**13. Defense Efficiency Behind Pitcher**  
| Weight: 6 | Data Source: Statcast (OAA/DRS) | Description: Rates team defensive range and conversion ability to adjust expected ERA and hit allowed props for ground ball or fly ball pitchers.  

**14. Clutch Leverage Performance**  
| Weight: 7 | Data Source: FanGraphs (WPA/LI splits) | Description: Evaluates high-leverage situational hitting (RISP, late/close) to weight RBI and run props differently based on game script probability.  

**15. Lineup Protection Context**  
| Weight: 7 | Data Source: FanGraphs (wOBA of surrounding hitters) | Description: Assesses quality of hitters batting immediately before and after to calculate intentional walk risk and RBI opportunity quality.  

**16. Rest/Travel Fatigue**  
| Weight: 6 | Data Source: MLB Schedule (Time zones, days rest) | Description: Flags cross-country travel, lack of rest days, or consecutive night games affecting reaction time and exit velocity.  

**17. Historical Head-to-Head**  
| Weight: 6 | Data Source: Baseball-Reference (PA vs. specific opponent) | Description: Analyzes career performance against specific pitchers or teams (min. 20 PA) to detect approach-based matchup edges not captured by models.  

**18. Injury/Health Proxy Flags**  
| Weight: 9 | Data Source: Statcast (SwStr%, Exit Velo drops) | Description: Detects hidden physical decline through sudden drops in swing speed, exit velocity, or pitch count efficiency before official injury reports.

============================================================
## SOCCER
============================================================

### TEAM PROFILE

Non-Penalty xG Trend | 10 | FBref/Understat | Rolling 5-game non-penalty expected goals per 90 versus season average isolating sustainable attacking quality independent of penalty variance.

High-Line Exposure Index | 8 | StatsBomb | Average defensive distance from goal in meters during last 5 fixtures calibrated against opponent attacking pace to measure transition vulnerability.

Set Piece xG Differential | 8 | FBref | Net expected goals generated from corners and free-kicks minus conceded over last 5 matches capturing dead-ball dominance in low-scoring fixtures.

Pressing Sustainability Decay | 9 | StatsBomb PPDA | Ratio of passes allowed per defensive action in minutes 0-30 versus 75-90 across last 5 games indicating fitness-based tactical drop-off.

Goalkeeper Shot-Stopping Form | 9 | FBref PSxG | Post-shot expected goals minus goals allowed per 90 over last 5 fixtures isolating keeper performance from defensive shot suppression.

Rotation Fatigue Variance | 7 | Transfermarkt | Coefficient of variation in minutes played by season-regular starters across

### PLAYER PROFILE

Goal Threat Coefficient | 10 | FBref/Understat | Composite of non-penalty xG, shots per 90, and big chances created to predict goal-scoring probability against specific defensive schemes.
Set Piece Dominance | 9 | Opta/WhoScored | Volume of penalties, direct free kicks, and corner delivery priority for direct goal contribution props.
Minutes Security Index | 10 | Lineup projections/Fitness reports | Probability of starting and projected 90s based on rotation risk, injury status, and tactical irreplaceability.
Defensive Vulnerability Matchup | 9 | Team-specific xG models | Opponent's defensive xG allowed to player's specific position, footedness, and movement patterns over last 5 matches.
Creative Impact Rating | 8 | Understat/StatsBomb | Expected assists, key passes, and progressive passes per 90 indicating assist probability and chance creation volume.
Card Risk Index | 7 | WhoScored/Discipline databases | Fouls committed per 90, tactical fouling tendency, and referee strictness rating for yellow/red card props.
Recent Form Trajectory | 8 | Rolling 5-match averages | Delta between season-long output and last 5 matches to detect upward/downward statistical trends.
Tactical Role Clarity | 8 | Heatmaps/Lineup data | Positional stability, set piece hierarchy, and penalty kick priority within the current formation.
Substitution Risk Profile | 8 | Manager patterns/Game state models | Likelihood of early withdrawal based on score state, rest protocols, and historical hook timing.
Direct Duel Dominance | 7 | Matchup history/FBref | Historical success rate in 1v1 situations and space creation against specific opponent defensive assignments.
Fixture Fatigue Factor | 7 | Fixture density/Travel logs | Days since last match, travel distance, and minutes accumulated in prior 14 days affecting explosive output capacity.
Clinical Finishing Variance | 8 | Understat/Post-shot xG | Delta between actual goals and expected goals indicating sustainable overperformance or regression risk.
Aerial Threat Coefficient | 7 | StatsBomb/FBref | Aerial duels won, headed shot volume, and target share for goal threat from crosses and set piece deliveries.
Pressing Resistance | 6 | StatsBomb/Opta | Ball retention under pressure and progressive carries per 90 to predict shot volume against high defensive lines.
Defensive Output Volume | 6 | FBref/Opta | Tackles, interceptions, and clearances per 90 for defensive action props and clean sheet correlation.
Home/Away Efficiency Delta | 6 | Season splits | Performance differential between home and away environments for goal/assist output props.
Motivation Context Multiplier | 5 | League tables/Fixture context | Statistical elevation or degradation in high-stakes scenarios versus dead rubber fixtures.
Weather/Condition Adaptability | 4 | Weather APIs/Historical splits | Output variance in extreme heat, cold, or poor pitch conditions versus optimal playing environments.

============================================================
## MMA
============================================================

### TEAM PROFILE

1. **Cardio Degradation Curve** | 9 | UFC Stats/Round-by-Round | Quantifies the percentage drop in significant strike output from round one to the final round across the last five fights to identify gas-tank liabilities before odds adjust.
2. **Weight Cut Reliability** | 9 | Athletic Commission Filings | Tracks weigh-in misses, hydration test failures, and subsequent round-one knockdown susceptibility over the last five bouts to flag depleted fighters.
3. **Damage Absorption Trajectory** | 9 | FightMetric | Trend line of significant head strikes absorbed per minute over the last five fights signaling chin degradation or defensive regression that casual bettors attribute to "bad luck."
4. **Strength of Schedule** | 8 | FightMatrix/Tapology | Aggregates opponent win percentages at the time of bout for the last five fights to deflate padded records against regional-level competition.
5. **Striking Efficiency Delta** | 8 | UFC Stats | Measures the differential between significant strikes landed and absorbed per minute across the last five bouts to expose fighters winning decisions while losing damage exchanges.
6. **Takedown Defense Trend** | 8 | UFC Stats | Percentage of takedown attempts stuffed in the last five fights versus career average to detect emerging wrestling vulnerabilities not yet priced into lines.
7. **Knockdown Recovery Rate** | 8 | Fight Video Analysis | Percentage of instances where the fighter was dropped or visibly rocked but recovered to win the round or fight in the last five outings indicating survival IQ under fire.
8. **Corner Adjustment Efficacy** | 7 | Fight Video/Audio | Binary scoring of tangible technical adjustments implemented between rounds in the last five bouts based on corner instruction quality and fighter execution.
9. **Camp Injury Density** | 7 | Media Reports/SCRAPP | Frequency of reported training injuries, sparring knockouts, or coach substitutions during the last five fight camps predicting withdrawal risk and compromised preparation.
10. **Grappling Control Dominance** | 7 | UFC Stats | Percentage of total fight time spent in dominant position per 15 minutes across the last five fights distinguishing suffocating top pressure from scrambly guard play.
11. **Pre-Fight Layoff Impact** | 7 | Fight History | Performance metric differential between fights following 9+ month layoffs versus back-to-back camps in the last five outings to model ring rust versus recovery benefits.
12. **Striking Defense Evolution** | 7 | FightMetric/ESPN | Head strike avoidance percentage trend across the last five fights specifically accounting for slips, rolls, and distance management rather than static blocking.
13. **Late Finish Probability** | 6 | UFC Stats | Ratio of round three or later finishes versus early finishes in the last five fights indicating the ability to conserve energy and execute in championship rounds.
14. **Weight Class Stability** | 6 | Commission Records | Number of weight class changes, radical cuts, or same-day weigh-in deltas over the last five fights correlating with performance volatility and hydration status.
15. **Altitude Adaptation Factor** | 6 | Fight History/Location | Win rate differential when fighting above 4,000 feet elevation or crossing three-plus time zones versus home region in the last five bouts accounting for hematocrit adjustment.
16. **Reach Utilization Efficiency** | 6 | UFC Stats | Jab landing percentage relative to reach advantage or disadvantage in the last five fights measuring disciplined distance management against shorter/longer opponents.
17. **Submission Threat Density** | 6 | UFC Stats | Submission attempts per 15 minutes of total ground time in the last five fights indicating active finishing danger versus control-oriented lay-and-pray wrestling.
18. **Judge Friendliness Index** | 5 | MMA Decisions | Percentage of split or majority decisions won versus lost in the last five fights adjusting for style bias toward volume strikers over damage dealers in scoring criteria.
19. **Clinch Exchange Efficiency** | 5 | UFC Stats | Significant strikes landed per minute in the clinch minus opponent's rate in the last five fights measuring dirty boxing and wall-walking superiority.
20. **Sparring Partner Quality** | 5 | Gym Affiliate Data | Aggregate winning percentage of primary sparring partners utilized during the last five camps serving as a proxy for preparation intensity and look-alike simulation quality.

### PLAYER PROFILE

1. **Striking Volume Output** | 9 | UFC Stats/FightMetric | Significant strikes landed per minute directly correlates with over-rounds and decision prop pricing.
2. **Grappling Control Dominance** | 9 | UFC Stats | Average control time per takedown drives under-rounds and submission-method probability.
3. **Knockout Threat Index** | 8 | UFC Stats/Tapology | Knockdowns per 15 minutes and KO/TKO finish rate for inside-distance method-of-victory odds.
4. **Submission Threat Index** | 8 | UFC Stats/Sherdog | Submission attempts per 15 minutes and finish rate for sub-method props and live-under potential.
5. **Takedown Defense Efficiency** | 8 | UFC Stats | Percentage of takedowns stuffed dictates fight locality and standing-strike prop viability.
6. **Cardio Regression Rate** | 9 | Round-by-round UFC Stats | Significant strike differential between round 1 and round 3+ for late-finish and over-rounds pricing.
7. **Chin Durability Score** | 7 | UFC Stats/Fight history | Knockdowns absorbed per fight and recovery success rate for fight-goes-distance markets.
8. **Weight Cut Reliability** | 6 | Athletic commissions/MMA Junkie | Historical weight misses and hydration failures indicating same-day performance degradation risk.
9. **Reach Differential Advantage** | 7 | UFC official measurements | Normalized arm/reach advantage over opponent for striking efficiency and output projections.
10. **Pace Pressure Index** | 8 | UFC Stats | Combined significant strikes and takedown attempts per minute as aggression proxy for total rounds volatility.
11. **Opponent Quality Rating** | 8 | Elo ratings/Tapology | Average opponent win percentage and UFC tenure for contextualizing raw stats against level of competition.
12. **Recent Form Trajectory** | 9 | Last 5 fights/Sherdog | Win/loss streak and performance trend for momentum-based prop adjustments and line movement.
13. **Ground Control Escape Rate** | 7 | UFC Stats | Successful bottom-position escapes per opportunity for fight-goes-distance probability in grappler matchups.
14. **Significant Strike Defense** | 8 | UFC Stats | Percentage of significant strikes avoided for durability assessment and opponent output ceiling projections.
15. **Fight IQ/Adaptability** | 6 | Video analysis/Corner audio | Mid-fight adjustment success and coach instruction adherence for live betting edge and late-round performance.
16. **Age-Adjusted Decline Curve** | 7 | Fight Matrix | Age relative to weight-class prime and fight-years for knockout vulnerability and cardio degradation modeling.
17. **Camp Quality Index** | 5 | Gym affiliations/Social media | Access to elite training partners and coaching infrastructure validating technical improvements.

============================================================
## BOXING
============================================================

### TEAM PROFILE

Recent Offensive Volume | 8 | CompuBox/FightMetrics | Average punches thrown per round over last 5 bouts normalized against divisional mean and career baseline.
Power Accuracy Premium | 9 | CompuBox | Ratio of power punches landed to thrown in last 5 fights versus career average, weighted by knockout conversion rate.
Defensive Evasion Rate | 8 | CompuBox/Computer Vision | Percentage of opponent punches avoided via head movement, footwork, or blocking in last 5 contests.
Chin Durability Index | 10 | BoxRec/Commission Reports | Composite score of knockdowns absorbed, recovery time from big hits, and stoppage history across career and last 5 fights.
Late Round Output Retention | 8 | Round-by-Round CompuBox | Percentage of punch volume and accuracy maintained in rounds 9-12 versus rounds 1-4 over last 5 championship distance fights.
Opposition Quality Rating | 9 | BoxRec/Elo Ratings | Aggregate win percentage and rating of last 5 opponents compared to career average competition level.
Momentum Trajectory | 9 | Fight Results/CompuBox | Slope of performance metrics (aggression, accuracy, defense) across last 5 fights indicating improvement or decline.
Reach Utilization Efficiency | 7 | CompuBox/Measurement Data | Jab landing rate and control time when fighting at maximum range versus closing distance in last 5 bouts.
Inside Fighting Competence | 7 | CompuBox/Clinch Analytics | Punch output and defensive efficiency when engaged at clinch distance including break speed and dirty boxing sanctions.
Weight Cut Stability | 8 | Commission Weigh-in/Hydration Tests | Hydration pass rate, weight class fluctuation history, and missed weight frequency over last 3 years.
Hand Speed Consistency | 7 | High-speed Video/CompuBox | Punches per minute and combination completion rate during exchanges in last 5 fights.
Cut/Physical Vulnerability | 8 | Fight Stoppage Records | Frequency of fights stopped due to cuts, hematomas, or doctor intervention in last 5 and career totals.
Corner Intervention Quality | 6 | Between-Round Metric Shifts | Swelling reduction efficiency and strategic adjustment success measured by round-to-round performance deltas.
Style-Specific Adaptation | 7 | Fight Database Analytics | Win rate and performance metrics against orthodox versus southpaw opponents in last 5 and career.
Significant Strike Absorption | 9 | CompuBox | Power punches landed on fighter per round, comparing last 5 trend to career average to detect defensive slippage.
KO Propensity Offensive | 9 | Fight Records/CompuBox | Knockdowns scored per power punch thrown ratio in last 5, plus first-half stoppage rate versus division average.
Referee Risk Profile | 5 | Commission Data | Point deductions, warnings, and disqualification frequency indicating foul-prone behavior patterns.
Layoff Rust Factor | 7 | Fight Dates/Training Reports | Inverse scoring based on months since last competitive round and camp continuity disruptions.

### PLAYER PROFILE

1. **One-Shot Power** | 10 | CompuBox/BoxStat KO% & Punch Velocity | Single-strike fight-ending probability driving Method of Victory and Round Group props.
2. **Chin Composite** | 9 | Fight footage/Knockdown history & Recovery time | Knockdown resistance and post-impact recovery capacity directly correlating with Distance/Over round viability.
3. **Cardio Trajectory** | 9 | Rounds 7-12 punch volume delta / Output decay rate | Late-round output retention determining Over/Under line edges and Decision prop confidence.
4. **Volume Consistency** | 8 | CompuBox punches thrown per round / Activity variance | Sustained offensive baseline for judge favorability and Decision victory probability.
5. **Defensive Evasion** | 8 | Punch stat % not landed / Head movement tracking | Clean punch absorption rate impacting Distance props and opponent connect percentage markets.
6. **Finish Conversion** | 8 | Knockdown-to-win ratio / Hunt mode efficiency | Ability to capitalize on hurt opponents for Exact Round and Live Method of Victory markets.
7. **Fight IQ Pivot** | 8 | Mid-fight momentum shift detection / Pattern break rate | Tactical adaptation speed affecting live betting volatility and round-specific momentum props.
8. **Activity Recency** | 8 | Days since last bout / Sparring footage reports | Ring rust coefficient and timing sharpness impacting early-round accuracy and Under props.
9. **Corner Efficacy** | 7 | Corner audio analysis / Between-round adjustment execution | Strategic modification quality affecting mid-fight momentum and live prop pricing.
10. **Weight Cut Stress** | 7 | Commission hydration tests / Rehydration weight delta | Same-day performance degradation risk for early-round Under and Knockdown props.
11. **Inside Fighting** | 7 | Clinch time / Body shots landed in phone booth | Dirty boxing durability affecting pace control and late-round fatigue markets.
12. **Body Attack Investment** | 7 | Body punch % / Cumulative damage correlation | Late-fade accumulation potential driving Round 7-12 stoppage props.
13. **Counter Accuracy** | 7 | Power counter connect % / First-strike defense | Reactive striking efficiency for sudden stoppage and flash knockdown probability.
14. **Hand Speed Burst** | 7 | Punch velocity data / Combination initiation rate | Beat-the-punch capability affecting early-round action props.
15. **Cut Vulnerability** | 6 | Commission medical history / Facial scar tissue mapping | Doctor stoppage likelihood affecting Fight to Go Distance and Round props.
16. **Reach Utilization** | 6 | Jab rate / Distance control maintenance | Physical attribute exploitation for fight flow and tactical Decision probability.
17. **Stance Asymmetry** | 5 | Southpaw proficiency vs Orthodox opponents / Lead hand accuracy | Matchup complexity coefficient for angle creation and line movement props.
18. **Clinch Control** | 6 | Clinch escape rate / Referee break dependency | Mauling resistance maintaining activity levels for volume-based Decision props.
ons and historical mid-fight improvement metrics.

**Punch Selection Variety** | 5 | Punch breakdown analytics | Diversity ratio between head, body, and uppercut targets preventing defensive adaptation and sustaining damage output.

============================================================
## TENNIS
============================================================

### TEAM PROFILE

**Tennis Team (Davis Cup/BJK Cup) Scouting Profile Criteria**

| Name | Weight | Data Source | Description |
|------|--------|-------------|-------------|
| **Second Serve Vulnerability Index** | 9 | ATP/WTA Tour Stats | Inverse of second serve points won percentage averaged across the squad's last 5 singles rubbers and season, predictive of break-point hemorrhaging under pressure. |
| **Injury Masking Indicator** | 10 | ATP/WTA Match Records + Medical Logs | Count of mid-match retirements, walkovers, or medical timeouts taken by roster players in last 10 individual matches, signaling imminent form collapse. |
| **Depth Beyond Anchor** | 9 | ATP/WTA Ranking Points | Combined singles ranking points of #2-#4 players versus opponent's 2-4 depth chart, weighted by recent 5-match win rates. |
| **Physical Fatigue Coefficient** | 9 | Tournament Logs + Flight Data | Cumulative court hours played by roster over last 14 days plus time zones crossed, adjusting for 5-set marathon recovery. |
| **Surface Transition Delta** | 8 | ATP/WTA Surface-Specific ELO | Performance variance between current tie surface and each player's optimal surface based on last 18 months of individual data. |
| **Clutch Break Point Conversion** | 8 | ATP/WTA Pressure Statistics | Aggregate percentage of break points converted when returning at 30-40 or deuce in deciding sets across last 5 team matches. |
| **Quality of Opposition Adjustment** | 8 | Universal Tennis Rating (UTR) + ELO | Weighted average strength of opponents faced in last 5 individual matches compared to season-average opponent quality. |
| **Doubles Pair Synergy Score** | 8 | ATP Doubles Stats + Davis Cup Historics | Chemistry metric derived from shared break-point conversion rates and first-return positioning efficiency in previous pairings. |
| **Deciding Set Reliability** | 8 | ITF Match Results | Team win percentage when rubbers reach a final set (or fifth set in doubles), filtered to last 5 competitive team events. |
| **Home Surface Manipulation** | 7 | Davis Cup/BJK Cup Historical Records | Win-rate delta when team selects surface (home ties) versus performance on neutral or away surfaces over last 3 years. |
| **Tiebreak Mastery Score** | 7 | ATP/WTA Tiebreak Records | Squad aggregate tiebreak win percentage in last 5 matches, isolating nerve performance in pressure neutral-game scenarios. |
| **Return Game Aggression Index** | 7 | ATP/WTA Return Stats | Percentage of opponent first-serve points won by the team's singles players, indicating ability to neutralize power servers. |
| **Altitude/Condition Adjustment** | 7 | Tournament Meteorological + Elevation Data | Historical performance differential of roster at current altitude (sea level vs. 1000m+) and indoor versus outdoor splits. |
| **Captain's Tactical Acumen** | 7 | ITF Lineup History | Historical success rate of reverse singles selections and doubles pairings in live fifth-rubber deciders. |
| **Momentum Decay Rate** | 7 | Set-by-Set Scoring Patterns | Frequency of immediately losing serve after breaking opponent (consolidation failure) in last 5 matches per player. |
| **First Serve Efficiency** | 6 | ATP/WTA Serve Statistics | First serve percentage in play multiplied by points won, averaged across roster's last 5 individual matches and season mean. |
| **Left-Handed Matchup Penalty** | 6 | Head-to-Head Records | Historical performance degradation when roster faces left-handed serves, adjusted for backhand vulnerability and wide ad-court exposure. |
| **Net Clearance Efficiency** | 6 | Hawk-Eye Tracking Data | Approach shot success rate minus unforced volley errors, relevant for fast-court doubles and singles net rushing. |
| **Weather Adaptability** | 6 | On-Court Condition Logs | Performance variance in high humidity (>70%) versus dry conditions affecting ball weight, string tension, and grip across last 5 outdoor matches. |

### PLAYER PROFILE

Hereâ€™s a focused list of 15-20 criteria for grading a playerâ€™s scouting profile, optimized for betting analysis and match impact. Each is weighted by importance (1-10) and includes data sources and a concise description:  

1. **Serve Rating (Weight: 9)** â€“ *ATP/WTA Stats, Match Charts*  
   Combines ace rate, 1st/2nd serve win%, and unreturned servesâ€”critical for hold games and tiebreaks.  

2. **Return Rating (Weight: 8)** â€“ *ATP/WTA Stats, Match Charts*  
   Measures break point conversion, return games won, and ability to pressure serves.  

3. **Recent Form (Weight: 7)** â€“ *Last 10 Matches, Elo Ratings*  
   Performance trend over recent matches (e.g., wins vs. top-50 players, tournament results).  

4. **Surface-Specific Win% (Weight: 8)** â€“ *Career/Season Stats*  
   Clay/grass/hard win ratesâ€”key for matchup edges (e.g., Nadal on clay).  

5. **Clutch Performance (Weight: 6)** â€“ *Tiebreak Win%, Deciding Sets*  
   Ability to win tight sets (3rd/5th sets) and pressure moments.  

6. **Fitness/Durability (Weight: 5)** â€“ *Retirement Stats, Match Lengths*  
   Risk of mid-match withdrawals or fatigue in long rallies/baseline battles.  

7. **Head-to-Head Record (Weight: 7)** â€“ *Historical Matchups*  
   Psychological/tactical edges against specific opponents (e.g., Djokovic vs. Federer).  

8. **Unforced Error Rate (Weight: 6)** â€“ *Match Stats*  
   Consistency under pressure; high errors hurt in long rallies.  

9. **Break Point Conversion (Weight: 7)** â€“ *ATP/WTA Stats*  
   % of break points wonâ€”directly impacts game/set betting markets.  

10. **Hold/Break Differential (Weight: 8)** â€“ *ATP/WTA Stats*  
    Net games won (holds minus breaks)â€”predicts dominance in a matchup.  

11. **First-Serve Percentage (Weight: 6)** â€“ *Match Stats*  
    High 1st-serve% reduces double faults and cheap points lost.  

12. **Net Play Efficiency (Weight: 4)** â€“ *Match Charts*  
    Volley/approach successâ€”matters most vs. baseliners or on fast surfaces.  

13. **Opponent-Specific Weaknesses (Weight: 6)** â€“ *Scouting Reports*  
    Exploitable flaws (e.g., poor backhand returns against lefty servers).  

14. **Tournament Performance (Weight: 5)** â€“ *Historical Results*  
    Past success at the event (e.g., Federer at Wimbledon).  

15. **Weather/Condition Adaptability (Weight: 4)** â€“ *Match Logs*  
    Wind/heat tolerance (e.g., Isnerâ€™s serve in indoor vs. outdoor).  

16. **Speed/Footwork (Weight: 5)** â€“ *Match Footage, Stats*  
    Defensive skills (e.g., retrieving drop shots on clay).  

17. **Mental Resilience (Weight: 6)** â€“ *Player Interviews, Comeback Wins*  
    Recovery from setbacks (e.g., losing first set but winning match).  

18. **Double Fault Frequency (Weight: 4)** â€“ *Match Stats*  
    High DF% hurts service games and total games props.  

19. **Average Rally Length (Weight: 5)** â€“ *Hawk-Eye Data*  
    Preference for short/long pointsâ€”impacts over/under betting.  

20. **Public Betting Sentiment (Weight: 3)** â€“ *Odds Movement*  
    Market bias (e.g., overvalued favorites) for contrarian opportunities.  

**Notes:**  
- Weights prioritize *direct* match/game impact (e.g., serve/return > net play).  
- Data sources lean on stats providers (ATP/WTA) and betting market signals.  
- Surface, H2H, and clutch stats are **multipliers** for matchup edges.  

Would you like adjustments for specific bet types (e.g., set handicaps, live betting)?

============================================================
## WNBA
============================================================

### TEAM PROFILE

_Not retrieved — model lacks sufficient data for this sport/type_

### PLAYER PROFILE

_Not retrieved — model lacks sufficient data for this sport/type_
