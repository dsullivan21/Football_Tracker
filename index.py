from pro_football_reference_web_scraper import player_game_log as p
from pro_football_reference_web_scraper import player_splits as ps
from pro_football_reference_web_scraper import team_splits as t
from pro_football_reference_web_scraper import team_game_log as tg
import pandas as pd
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
import numpy as np
import scipy.stats
from scipy import stats
from sklearn import datasets, svm
from sklearn.linear_model import LinearRegression
import PySimpleGUI as sg
import json
from numpy.polynomial.polynomial import polyfit
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sg.theme('BluePurple')



# ---- Layout Section ----- #
game_log =""
playerName = ""

all_projections = []

#weight of each for projections
weights = []

#hold all data for projections
allData = []

#Get the FPI data for weeks W/L Chances
with open('chancetowin.json', 'r') as f:
    fpi_values = json.load(f)

#Open rosters sheet
with open('rosters.json', 'r') as f:
    rosters = json.load(f)

#Get Win/Loss Splits data projection 
# --------------- NO REGRESSION WITH THIS ----------- #
def win_loss_calc (team_fpi, opp_fpi, player_wl_splits, playerpos):

    team_fpi = team_fpi/100
    opp_fpi = opp_fpi/100

    #return object
    arr = {}
    if playerpos == "RB":
        w_ruattemps = player_wl_splits.loc['W']['rush_att']
        w_ruyards = player_wl_splits.loc['W']['rush_yds']
        w_rutds = player_wl_splits.loc['W']['rush_td']
        w_targets = player_wl_splits.loc['W']['tgt']
        w_recyards = player_wl_splits.loc['W']['rec_yds']
        w_rectds = player_wl_splits.loc['W']['rec_td']

        l_ruattemps = player_wl_splits.loc['L']['rush_att']
        l_ruyards = player_wl_splits.loc['L']['rush_yds']
        l_rutds = player_wl_splits.loc['L']['rush_td']
        l_targets = player_wl_splits.loc['L']['tgt']
        l_recyards = player_wl_splits.loc['L']['rec_yds']
        l_rectds = player_wl_splits.loc['L']['rec_td']

        totalatt = (w_ruattemps *team_fpi) + (l_ruattemps * opp_fpi)
        totalruyards = (w_ruyards *team_fpi) + (l_ruyards * opp_fpi)
        totaltarg = (w_targets *team_fpi) + (l_targets * opp_fpi)
        totalrecyards = (w_recyards *team_fpi) + (l_recyards * opp_fpi)
        totaltds = (w_rutds *team_fpi) + (l_rutds * opp_fpi)
        totalrectds = (w_rectds *team_fpi) + (l_rectds * opp_fpi)

        arr = { 
            "recent_projection": {
                "rush_attempts": totalatt,
                "rush_yards": totalruyards,
                "targets": totaltarg,
                "rec_yards": totalrecyards,
                "rush_tds": totaltds,
                "rec_tds": totalrectds,
                "weight" : 0.2
            },
        }
    elif playerpos == "WR" or playerpos == "TE":
        w_rec = player_wl_splits.loc['W']['rec']
        w_snap = player_wl_splits.loc['W']['snap_pct']
        w_targets = player_wl_splits.loc['W']['tgt']
        w_recyards = player_wl_splits.loc['W']['rec_yds']
        w_rectds = player_wl_splits.loc['W']['rec_td']

        l_rec = player_wl_splits.loc['L']['rec']
        l_snap = player_wl_splits.loc['L']['snap_pct']
        l_targets = player_wl_splits.loc['L']['tgt']
        l_recyards = player_wl_splits.loc['L']['rec_yds']
        l_rectds = player_wl_splits.loc['L']['rec_td']

        totalrec = (w_rec *team_fpi) + (l_rec * opp_fpi)
        totalsnap = (w_snap *team_fpi) + (l_snap * opp_fpi)
        totaltarg = (w_targets *team_fpi) + (l_targets * opp_fpi)
        totalrecyards = (w_recyards *team_fpi) + (l_recyards * opp_fpi)
        totalrectds = (w_rectds *team_fpi) + (l_rectds * opp_fpi)

        arr = { 
            "recent_projection":{
                "rec": totalrec,
                "snap_pct": totalsnap,
                "targets": totaltarg,
                "rec_yards": totalrecyards,
                "rec_tds": totalrectds,
                "weight" : 0.15
            },
        }
        print("WL SPLIT: ", arr)

    elif playerpos == "QB":

        w_ruattemps = player_wl_splits.loc['W']['rush_att']
        w_ruyards = player_wl_splits.loc['W']['rush_yds']
        w_rutds = player_wl_splits.loc['W']['rush_td']
        w_att = player_wl_splits.loc['W']['att']
        w_cmp = player_wl_splits.loc['W']['cmp']
        w_pass_yards = player_wl_splits.loc['W']['pass_yds']
        w_pass_tds = player_wl_splits.loc['W']['pass_td']
        w_pass_ints = player_wl_splits.loc['W']['int']

        l_ruattemps = player_wl_splits.loc['L']['rush_att']
        l_ruyards = player_wl_splits.loc['L']['rush_yds']
        l_rutds = player_wl_splits.loc['L']['rush_td']
        l_att = player_wl_splits.loc['L']['att']
        l_cmp = player_wl_splits.loc['L']['cmp']
        l_pass_yards = player_wl_splits.loc['L']['pass_yds']
        l_pass_tds = player_wl_splits.loc['L']['pass_td']
        l_pass_ints = player_wl_splits.loc['L']['int']

        totalruatt = (w_ruattemps *team_fpi) + (l_ruattemps * opp_fpi)
        totalatt = (w_att *team_fpi) + (w_att * opp_fpi)
        totalruyards = (w_ruyards *team_fpi) + (l_ruyards * opp_fpi)
        totalcmp = (w_cmp *team_fpi) + (l_cmp * opp_fpi)
        totalpassyards = (w_pass_yards *team_fpi) + (l_pass_yards * opp_fpi)
        totaltds = (w_rutds *team_fpi) + (l_rutds * opp_fpi)
        totalpasstds = (w_pass_tds *team_fpi) + (l_pass_tds * opp_fpi)
        totalints = (w_pass_ints *team_fpi) + (l_pass_ints * opp_fpi)
        
        arr = {
            "recent_projection":{
                "rush_attempts": totalruatt,
                "rush_yards": totalruyards,
                "rush_tds": totaltds,
                "att": totalatt,
                "cmp": totalcmp,
                "int": totalints,
                "pass_yards": totalpassyards,
                "pass_tds": totalpasstds,
                "weight": 0.15
            },
        }


    return arr


#calculate regression
# return a,b in solution to y = ax + b such that root mean square distance between trend line and original points is minimized

def linReg(X, Y):
    N = len(X)
    Sx = Sy = Sxx = Syy = Sxy = 0.0
    for x, y in zip(X, Y):
        Sx = Sx + x
        Sy = Sy + y
        Sxx = Sxx + x*x
        Syy = Syy + y*y
        Sxy = Sxy + x*y
    det = Sxx * N - Sx * Sx
    return (Sxy * N - Sy * Sx)/det, (Sxx * Sy - Sx * Sxy)/det


# value for projection, yvalues, xvalues returns projection
def linearRegPredict(predictVal, xval, yval):

    projection_X = 5
    length = len(np.array(xval))
  #  xval2 = xval.tolist()
  #  xval2.pop(0)
  #  xval2 = np.array(xval2)
  #  X = preprocessing.scale(xval2)
  #  X = X.reshape(8, 2)
 #   print("PREPROCCESS: ", X)
 #   yval2 = np.array([1,2,3,4,5,6,7,8])
  #  print(yval2)
  #  X_train, X_test, y_train, y_test = train_test_split(np.array(X), yval2, test_size=0.6)
  #  lr = LinearRegression(n_jobs=-1)
  #  lr.fit(X_train, y_train)
  #  confidence = lr.score(X_test, y_test)
  #  predic = lr.predict(np.array(X))
  #  print("Predict", predic)
  #  print("X, Y: ", X_test, y_test)
  #  print("CONFIDENCE: ", confidence)

    #Project Based on trend
    if predictVal > 0.25 and predictVal < 0.5:
        projection_X = length * 0.58
    elif predictVal > 0.5 and predictVal < 0.75:
        projection_X = length * 0.8
    elif predictVal > 0.75:
        projection_X = length *0.9
    elif predictVal <0.25 and predictVal > 0.1:
        projection_X = length * 0.585
    elif predictVal < 0.1 and predictVal >= 0:
        projection_X = length * 0.49
    elif predictVal < 0 and predictVal > -0.1:
        projection_X = length * .55
    elif predictVal < -0.25 and predictVal > -0.5:
        projection_X = length * .6
    elif predictVal < -0.5:
        projection_X = length * 0.72

    projected_value = 0
    lr = LinearRegression()
    lr.fit(np.array(yval).reshape(-1,1), np.array(xval).reshape(-1,1))
    projected_value = lr.predict(np.array([projection_X]).reshape(-1,1))
    print("Coeff: ", predictVal, "\nProjection x val: ", projection_X, "\nProjection: ", projected_value)

    return projected_value

# plot values and projection
def plotGraph(player_game_log, colname, ppos, loglast):

    if ppos == "WR" or ppos == "TE":
        if (len(loglast) > 0):
            df = player_game_log[['date','week','team','game_location','opp','result','team_pts','opp_pts','tgt','rec','rec_yds','rec_td','snap_pct']]
            dflast = loglast[['date','week','team','game_location','opp','result','team_pts','opp_pts','tgt','rec','rec_yds','rec_td','snap_pct']]
            df = pd.concat([dflast, df], ignore_index=True)
        else:
            df = player_game_log[['date','week','team','game_location','opp','result','team_pts','opp_pts','tgt','rec','rec_yds','rec_td','snap_pct']]
    elif ppos == "RB":
        if (len(loglast)> 0):
            df = player_game_log[['date','week','team','game_location','opp','result','team_pts','opp_pts','rush_att','rush_yds','rush_td','tgt','rec_yds', 'rec_td']]
            dflast = loglast[['date','week','team','game_location','opp','result','team_pts','opp_pts','rush_att','rush_yds','rush_td','tgt','rec_yds', 'rec_td']]
            df = pd.concat([dflast, df], ignore_index=True)
        else: 
            df = player_game_log[['date','week','team','game_location','opp','result','team_pts','opp_pts','rush_att','rush_yds','rush_td','tgt','rec_yds', 'rec_td']]
    elif ppos == "QB":
        if loglast is None:
            df = player_game_log[['date','week','team','game_location','opp','result','team_pts','opp_pts','cmp', 'att', 'pass_yds', 'pass_td','int','rating','sacked','rush_att','rush_yds','rush_td']]
        if (len(loglast) > 0 ):
            df = player_game_log[['date','week','team','game_location','opp','result','team_pts','opp_pts','cmp', 'att', 'pass_yds', 'pass_td','int','rating','sacked','rush_att','rush_yds','rush_td']]
            dflast = loglast[['date','week','team','game_location','opp','result','team_pts','opp_pts','cmp', 'att', 'pass_yds', 'pass_td','int','rating','sacked','rush_att','rush_yds','rush_td']]
            df = pd.concat([dflast, df], ignore_index=True)
        else:
            df = player_game_log[['date','week','team','game_location','opp','result','team_pts','opp_pts','cmp', 'att', 'pass_yds', 'pass_td','int','rating','sacked','rush_att','rush_yds','rush_td']]
            
    forecast_col = colname
    forecast_out = int(math.ceil(0.01 * len(df)))
    print("out: ", forecast_out)
    df['label'] = df[forecast_col]

    print(df)
    print("DF LABEL: ", df['label'])
    df2 = df['label']
    X = np.array(range(len(df2))).reshape(-1, 1)

    print(X)
    X = preprocessing.scale(X)
    X_lately = X[-forecast_out:]
    #X = X[:-forecast_out]
    print("LATELY", X_lately)

    df.dropna(inplace=True)

    y = np.array(df['label'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    clf = LinearRegression(n_jobs=-1)
    clf.fit(X_train, y_train)
    print( X_train, X_test, y_train, y_test)
    confidence = clf.score(X_test, y_test)

    print(confidence)
    forecast_set = clf.predict(X_lately)
    print("FORECAST SET", forecast_set)
    df['Forecast'] = np.nan

    last_date = df.iloc[-1].name
    print(last_date)
    next_date = last_date+1
    for i in forecast_set:
        df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]
        next_date += 1
        print(next_date)


    return forecast_set

#player_game_log = p.get_player_game_log(player = "David Njoku", position = "TE", season = 2022)
#plotGraph(player_game_log, "rec_yds")
#confidence interval function
def mean_confidence_interval(data, confidence=0.8):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return [m, m-h, m+h]

def runRbProj(playerFirst, playerLast, pposition, oppTeam,oppTeamABR,loc, playerfull):
    # ------ Load Player data for Model ----- #
    if playerfull == "":
        playerName = playerFirst + " " + playerLast
    else:
        playerName =playerfull

    prevGames = []
    allprevGames = []
    #Save opponent Win Probability Based on FPI
    oppWinProb = fpi_values[oppTeamABR]
    #init team win 
    teamWinProb = 0
    #init Player Team
    playerTeam = ""

    # ------ Load Opp Team Data for Model -------- #

    opp_games = tg.get_team_game_log(team = oppTeam, season = 2023)
    opp_splits = t.home_road(team = oppTeam, season = 2022, avg = True)

    #Store Opposition Defense statistics 
    opp_rushing_defense = opp_games.loc[:, 'opp_rush_yds'].values
    opp_ru_defense_in_wins = opp_games[(opp_games["result"] == "W")]['opp_rush_yds'].values
    opp_ru_defense_in_loss = opp_games[(opp_games["result"] == "L")]['opp_rush_yds'].values

    rushing_sum = sum(opp_rushing_defense)

    rushing_average = rushing_sum/len(opp_rushing_defense)

    yval_wins = range(len(opp_ru_defense_in_wins))
    yval_loss = range(len(opp_ru_defense_in_loss))
    yval = range(len(opp_rushing_defense))
    val = linReg(yval, opp_rushing_defense)[0]

    defensive_ru_allowed = linearRegPredict(val, opp_rushing_defense, yval)[0][0]

    print("Defensive Projection: ", defensive_ru_allowed)
    print("Defensive Average: ", rushing_average)

    #get defensive rushing projection coeff

    ru_d_av_w = sum(opp_ru_defense_in_wins)/len(opp_ru_defense_in_wins)
    ru_d_av_l = sum(opp_ru_defense_in_loss)/len(opp_ru_defense_in_loss)

    rushing_coeff = defensive_ru_allowed/rushing_average

    # ---- Get ready to store historical data ---- #
    historical_rushatt = []
    historical_rushyrds = []
    historical_rushtds = []
    historical_targets = []
    historical_recyrds = []
    historical_rectds = []

    recent_games_rushatt=[]
    recent_games_rushyrds=[]
    recent_games_rushtds=[]
    recent_games_targets=[]
    recent_games_recyrds=[]
    recent_games_rectds=[]
    forecast_pick = -1
    dfLive = True

    log2023 = ""
    log2022 = ""

    i = 2024
    #search through last 4 years data
    while (i >= 2019):
    
        try:
            player_game_log = p.get_player_game_log(player = playerName, position = pposition, season = i)
            dfLive = True
        except:             
            print("player didnt return a result")
            dfLive = False

        # ---- Get data from all games for recent games trend ---- #
        if i == 2024 and dfLive == True:
            playerTeam = player_game_log.loc[0]['team']
            recent_games_rushatt.extend( player_game_log.loc[:,'rush_att'].values)
            recent_games_rushyrds.extend(player_game_log.loc[:,'rush_yds'].values)
            recent_games_rushtds.extend(player_game_log.loc[:,'rush_td'].values)
            recent_games_targets.extend(player_game_log.loc[:,'tgt'].values)
            recent_games_recyrds.extend(player_game_log.loc[:,'rec_yds'].values)
            recent_games_rectds.extend(player_game_log.loc[:,'rec_td'].values)
            log2023 = player_game_log
        elif i == 2023 and dfLive == True:
            recent_games_rushatt = player_game_log.loc[:,'rush_att'].values
            recent_games_rushyrds = player_game_log.loc[:,'rush_yds'].values
            recent_games_rushtds = player_game_log.loc[:,'rush_td'].values
            recent_games_targets = player_game_log.loc[:,'tgt'].values
            recent_games_recyrds = player_game_log.loc[:,'rec_yds'].values
            recent_games_rectds = player_game_log.loc[:,'rec_td'].values
            historical_rectds.extend(player_game_log.loc[:,'rec_td'].values)
            historical_recyrds.extend(player_game_log.loc[:,'rec_yds'].values)
            historical_rushyrds.extend(player_game_log.loc[:,'rush_yds'].values)
            historical_rushtds.extend(player_game_log.loc[:,'rush_td'].values)
            historical_rushatt.extend(player_game_log.loc[:,'rush_att'].values)
            historical_targets.extend( player_game_log.loc[:,'tgt'].values)
            log2022 = player_game_log
            # Get player team
            if playerTeam == "":
                playerTeam = player_game_log.loc[0]['team']
        elif dfLive == True:
            historical_rectds.extend(player_game_log.loc[:,'rec_td'].values)
            historical_recyrds.extend(player_game_log.loc[:,'rec_yds'].values)
            historical_rushyrds.extend(player_game_log.loc[:,'rush_yds'].values)
            historical_rushtds.extend(player_game_log.loc[:,'rush_td'].values)
            historical_rushatt.extend(player_game_log.loc[:,'rush_att'].values)
            historical_targets.extend( player_game_log.loc[:,'tgt'].values)
        
        # --------------------------------
        # TODO: Test if this If is working
        # --------------------------------

        isGame = False

        if loc == "Away" :
        # gets data for previous years similar game if available 
            valRow = player_game_log[(player_game_log["game_location"] == "@") & (player_game_log["opp"] == oppTeamABR)]
            print(valRow)
            if len(valRow.index) > 0:
                ruattemps = valRow.iloc[0]['rush_att']
                ruyards = valRow.iloc[0]['rush_yds']
                rutds = valRow.iloc[0]['rush_td']
                targets = valRow.iloc[0]['tgt']
                recyards = valRow.iloc[0]['rec_yds']
                rectds = valRow.iloc[0]['rec_td']
                ruattemps_all = valRow.iloc[0]['rush_att']
                ruyards_all = valRow.iloc[0]['rush_yds']
                rutds_all = valRow.iloc[0]['rush_td']
                targets_all = valRow.iloc[0]['tgt']
                recyards_all = valRow.iloc[0]['rec_yds']
                rectds_all = valRow.iloc[0]['rec_td']
                isGame = True
        elif loc == "Home":
            valRow = player_game_log[(player_game_log["game_location"] != "@") & (player_game_log["opp"] == oppTeamABR)]
            print(valRow)
            if len(valRow.index) > 0:
                ruattemps = valRow.iloc[0]['rush_att']
                ruyards = valRow.iloc[0]['rush_yds']
                rutds = valRow.iloc[0]['rush_td']
                targets = valRow.iloc[0]['tgt']
                recyards = valRow.iloc[0]['rec_yds']
                rectds = valRow.iloc[0]['rec_td']
                ruattemps_all = valRow.iloc[0]['rush_att']
                ruyards_all = valRow.iloc[0]['rush_yds']
                rutds_all = valRow.iloc[0]['rush_td']
                targets_all = valRow.iloc[0]['tgt']
                recyards_all = valRow.iloc[0]['rec_yds']
                rectds_all = valRow.iloc[0]['rec_td']
                isGame = True

        print(isGame)
        # Store game data
        if isGame:
            gamedata = {
                "rush_attempts": ruattemps,
                "rush_yards": ruyards,
                "rush_tds": rutds,
                "targets": targets,
                "rec_yards": recyards,
                "rec_tds": rectds,
                "year": i
            }
            prevGames.append(gamedata)
            all_pre_gamedata = {
                "rush_attempts": ruattemps_all,
                "rush_yards": ruyards_all,
                "rush_tds": rutds_all,
                "targets": targets_all,
                "rec_yards": recyards_all,
                "rec_tds": rectds_all,
                "year": i
            }
            allprevGames.append(all_pre_gamedata)
        i = i -1

        #END WHILE#

    prds = plotGraph(log2023, "rush_yds", "RB", log2022)
    precyds = plotGraph(log2023, "rec_yds", "RB", log2022)

        #---------- CALCULATE HISTORICAL MODEL ----------#
    ru_allowed = (ru_d_av_w * (oppWinProb/100)) + (ru_d_av_l * (fpi_values[playerTeam]/100))

    pass_val = ru_allowed - rushing_average
    new_val = pass_val/rushing_average 
    print("rushing diff: ", pass_val)
    print ("to add to coeff " ,new_val)
    rushing_coeff = rushing_coeff + new_val

    #Get Last season splits based on W/L
    player_w_l_splits = ps.win_loss(player = playerName, position = pposition, season = 2022, avg = True)
    teamWinProb = fpi_values[playerTeam]

    #calculate W\L changes .2 WEIGHT
    win_loss_split = win_loss_calc(teamWinProb, oppWinProb, player_w_l_splits, pposition)

    print("TEAM WIN : ", teamWinProb, "OPP WIN: ", oppWinProb)
    #rush yards
    yvalues = range(len(historical_rushyrds))
    val = linReg(yvalues, historical_rushyrds)[0]
    historical_ruyards_proj = linearRegPredict(val, historical_rushyrds, yvalues)
    historical_ruyards_proj = historical_ruyards_proj[0][0]
    #rec yards
    yvalues = range(len(historical_recyrds))
    val = linReg(yvalues, historical_recyrds)[0]
    historical_recyards_proj = linearRegPredict(val, historical_recyrds, yvalues)
    historical_recyards_proj = historical_recyards_proj[0][0]
    #recTds
    yvalues = range(len(historical_rectds))
    val = linReg(yvalues, historical_rectds)[0]
    historical_rectds_proj = linearRegPredict(val, historical_rectds, yvalues)
    historical_rectds_proj = historical_rectds_proj[0][0]
    #rushTds
    yvalues = range(len(historical_rushtds))
    val = linReg(yvalues, historical_rushtds)[0]
    historical_rutds_proj = linearRegPredict(val, historical_rushtds, yvalues)
    historical_rutds_proj =historical_rutds_proj[0][0]
    #rushatt
    yvalues = range(len(historical_rushatt))
    val = linReg(yvalues, historical_rushatt)[0]
    historical_ruatt_proj = linearRegPredict(val, historical_rushatt, yvalues)
    historical_ruatt_proj= historical_ruatt_proj[0][0]
    #targets
    yvalues = range(len(historical_targets))
    val = linReg(yvalues, historical_targets)[0]
    historical_targets_proj = linearRegPredict(val, historical_targets, yvalues)
    historical_targets_proj =historical_targets_proj[0][0]

    print("historical_ruyards_proj: ", historical_ruyards_proj, " historical_recyards_proj ", historical_recyards_proj, " historical_rectds_proj ", historical_rectds_proj, " historical_rutds_proj ",historical_rutds_proj, " historical_ruatt_proj ", historical_ruatt_proj , " historical_targets_proj ", historical_targets_proj  )

    #---------- CALCULATE RECENT GAMES MODEL ----------#

    #rush yards
    yvalues = range(len(recent_games_rushyrds))
    val = linReg(yvalues, recent_games_rushyrds)[0]
    recent_games_rushyrds_proj = linearRegPredict(val, recent_games_rushyrds, yvalues)
    recent_games_rushyrds_proj = recent_games_rushyrds_proj[0][0]
    #rec yards
    yvalues = range(len(recent_games_recyrds))
    val = linReg(yvalues, recent_games_recyrds)[0]
    recent_games_recyrds_proj = linearRegPredict(val, recent_games_recyrds, yvalues)
    recent_games_recyrds_proj = recent_games_recyrds_proj[0][0]
    #recTds
    yvalues = range(len(recent_games_rectds))
    val = linReg(yvalues, recent_games_rectds)[0]
    recent_games_rectds_proj = linearRegPredict(val, recent_games_rectds, yvalues)
    recent_games_rectds_proj = recent_games_rectds_proj[0][0]
    #rushTds
    yvalues = range(len(recent_games_rushtds))
    val = linReg(yvalues, recent_games_rushtds)[0]
    recent_games_rushtds_proj = linearRegPredict(val, recent_games_rushtds, yvalues)
    recent_games_rushtds_proj =recent_games_rushtds_proj[0][0]
    #rushatt
    yvalues = range(len(recent_games_rushatt))
    val = linReg(yvalues, recent_games_rushatt)[0]
    recent_games_rushatt_proj = linearRegPredict(val, recent_games_rushatt, yvalues)
    recent_games_rushatt_proj= recent_games_rushatt_proj[0][0]
    #targets
    yvalues = range(len(recent_games_targets))
    val = linReg(yvalues, recent_games_targets)[0]
    recent_games_targets_proj = linearRegPredict(val, recent_games_targets, yvalues)
    recent_games_targets_proj =recent_games_targets_proj[0][0]

    totalruyards = 0
    totaltds = 0
    totalatt = 0
    totaltarg = 0
    totalrecyards = 0
    totalrectds = 0

    count = 0
    yval = []
    yardTrend = []
    attTrend = []
    rutdTrend = []
    rectdTrend = []
    targTrend = []
    recyrdTrend = []

    confidence = mean_confidence_interval(recent_games_rushyrds)

    #all Prev Games
    if len(allprevGames) > 1:

        for game in allprevGames:
            totalruyards = totalruyards + game['rush_yards']
            totaltds = totaltds + game['rush_tds']
            totalatt = totalatt + game['rush_attempts']
            totaltarg = totaltarg + game['targets']
            totalrectds = totalrectds +game['rec_tds']
            totalrecyards = totalrecyards + game['rec_yards']
            count = count + 1
            print("Ru Att: " , game['rush_attempts'], " Year: ", game['year'] )
            yval.append(count)
            yardTrend.append (game['rush_yards'])
            attTrend.append (game['rush_attempts'])
            rutdTrend.append(game['rush_tds'])
            rectdTrend.append(game['rec_tds'])
            targTrend.append(game['targets'])
            recyrdTrend.append(game['rec_yards'])
        #rush yards
        val = linReg(yval, yardTrend)[0]
        yardregression = linearRegPredict(val, yardTrend, yval)
        yardregression = yardregression[0][0]
        #rec yards
        val = linReg(yvalues, recyrdTrend)[0]
        recyardregression = linearRegPredict(val, recyrdTrend, yval)
        recyardregression = recyardregression[0][0]
        #recTds
        val = linReg(yvalues, rectdTrend)[0]
        rectdregression = linearRegPredict(val, rectdTrend, yval)
        rectdregression = rectdregression[0][0]
        #rushTds
        val = linReg(yvalues, rutdTrend)[0]
        rutdregression = linearRegPredict(val, rutdTrend, yval)
        rutdregression =rutdregression[0][0]
        #rushatt
        val = linReg(yval, attTrend)[0]
        attregression = linearRegPredict(val, attTrend, yval)
        attregression= attregression[0][0]
        #targets
        val = linReg(yvalues, targTrend)[0]
        targregression = linearRegPredict(val, targTrend, yval)
        targregression =targregression[0][0]

        # print(attregression)
        # print (tdRegression)

        # --- Calc averages from previous 4 games --- #
        totalruyards = totalruyards/count
        totaltds = totaltds/count
        totalatt = totalatt/count
        totaltarg = totaltarg/count
        totalrecyards = totalrecyards/count
        totalrectds = totalrectds/count

        all_avgs = { "recent_projection": {
            "rush_attempts": totalatt,
            "rush_yards": totalruyards,
            "targets": totaltarg,
            "rec_yards": totalrecyards,
            "rush_tds": totaltds,
            "rec_tds": totalrectds,
            "weight" : 0.1
            },
        }
        all_reg_avgs = { "recent_projection": {
            "rush_attempts": attregression,
            "rush_yards": yardregression,
            "targets": targregression,
            "rec_yards": recyardregression,
            "rush_tds": rutdregression,
            "rec_tds": rectdregression,
            "weight" : 0.1
            },
        }

    totalruyards = 0
    totaltds = 0
    totalatt = 0
    totaltarg = 0
    totalrecyards = 0
    totalrectds = 0

    count = 0
    yval = []
    yardTrend = []
    attTrend = []
    rutdTrend = []
    rectdTrend = []
    targTrend = []
    recyrdTrend = []

    recent_confidence = mean_confidence_interval(recent_games_rushyrds)
    historical_confidence = mean_confidence_interval(historical_rushyrds)

    confidence = [(recent_confidence[0]*0.75) + (historical_confidence[0]*0.25), (recent_confidence[1]*0.75) + (historical_confidence[1]*0.25), (recent_confidence[2]*0.75) + (historical_confidence[2]*0.25)]

    for game in prevGames:

        totalruyards = totalruyards + game['rush_yards']
        totaltds = totaltds + game['rush_tds']
        totalatt = totalatt + game['rush_attempts']
        totaltarg = totaltarg + game['targets']
        totalrectds = totalrectds +game['rec_tds']
        totalrecyards = totalrecyards + game['rec_yards']
        count = count + 1
        print("Ru Att: " , game['rush_attempts'], " Year: ", game['year'] )
        yval.append(count)
        yardTrend.append (game['rush_yards'])
        attTrend.append (game['rush_attempts'])
        rutdTrend.append(game['rush_tds'])
        rectdTrend.append(game['rec_tds'])
        targTrend.append(game['targets'])
        recyrdTrend.append(game['rec_yards'])

    # -- Calc Prev game Regression -- #

    # -- Check trend value -- #

    # rutrend = linReg(yval, yardTrend)
    #  rutrend = rutrend[0]

    if count > 1: 
        #rush yards
        val = linReg(yval, yardTrend)[0]
        yardregression = linearRegPredict(val, yardTrend, yval)
        yardregression = yardregression[0][0]
        #rec yards
        val = linReg(yvalues, recyrdTrend)[0]
        recyardregression = linearRegPredict(val, recyrdTrend, yval)
        recyardregression = recyardregression[0][0]
        #recTds
        val = linReg(yvalues, rectdTrend)[0]
        rectdregression = linearRegPredict(val, rectdTrend, yval)
        rectdregression = rectdregression[0][0]
        #rushTds
        val = linReg(yvalues, rutdTrend)[0]
        rutdregression = linearRegPredict(val, rutdTrend, yval)
        rutdregression =rutdregression[0][0]
        #rushatt
        val = linReg(yval, attTrend)[0]
        attregression = linearRegPredict(val, attTrend, yval)
        attregression= attregression[0][0]
        #targets
        val = linReg(yvalues, targTrend)[0]
        targregression = linearRegPredict(val, targTrend, yval)
        targregression =targregression[0][0]

        # print(attregression)
        # print (tdRegression)

        # --- Calc averages from previous 4 games --- #
        totalruyards = totalruyards/count
        totaltds = totaltds/count
        totalatt = totalatt/count
        totaltarg = totaltarg/count
        totalrecyards = totalrecyards/count
        totalrectds = totalrectds/count

    if count > 1 :
        # ----- Previous Similar Games ----- #
        avgs = { "recent_projection": {
            "rush_attempts": totalatt,
            "rush_yards": totalruyards,
            "targets": totaltarg,
            "rec_yards": totalrecyards,
            "rush_tds": totaltds,
            "rec_tds": totalrectds,
            "weight" : 0.05
            },
        }

        # ------ Projection from previous games ----- #
        reg_avgs = { "recent_projection": {
            "rush_attempts": attregression,
            "rush_yards": yardregression,
            "targets": targregression,
            "rec_yards": recyardregression,
            "rush_tds": rutdregression,
            "rec_tds": rectdregression,
            "weight" : 0.1
            },
        }

        # ------ Historical Projections -------- #
        hist_avg = { 
            "recent_projection": {
            "rush_attempts": historical_ruatt_proj,
            "rush_yards": historical_ruyards_proj,
            "targets": historical_targets_proj,
            "rec_yards": historical_recyards_proj,
            "rush_tds": historical_rutds_proj,
            "rec_tds": historical_rectds_proj,
            "weight" : 0.25
            },
        }
    
        # ------ Recent Game Projections -------- #
        recent_avg = { 
            "recent_projection": {
            "rush_attempts": recent_games_rushatt_proj,
            "rush_yards": recent_games_rushyrds_proj,
            "targets": recent_games_targets_proj,
            "rec_yards": recent_games_recyrds_proj,
            "rush_tds": recent_games_rushtds_proj,
            "rec_tds": recent_games_rectds_proj,
            "weight" : 0.2
            },
        }
    elif count < 1 and len(allprevGames) > 1:
        # ----- Previous Similar Games ----- #
        avgs = { "recent_projection": {
            "rush_attempts": 0,
            "rush_yards": 0,
            "targets": 0,
            "rec_yards": 0,
            "rush_tds": 0,
            "rec_tds": 0,
            "weight" : 0
            },
        }

        # ------ Projection from previous games ----- #
        reg_avgs = { "recent_projection": {
            "rush_attempts": 0,
            "rush_yards": 0,
            "targets": 0,
            "rec_yards": 0,
            "rush_tds": 0,
            "rec_tds": 0,
            "weight" : 0
            },
        }

        # ------ Historical Projections -------- #
        hist_avg = { 
            "recent_projection": {
            "rush_attempts": historical_ruatt_proj,
            "rush_yards": historical_ruyards_proj,
            "targets": historical_targets_proj,
            "rec_yards": historical_recyards_proj,
            "rush_tds": historical_rutds_proj,
            "rec_tds": historical_rectds_proj,
            "weight" : 0.2
            },
        }
    
        # ------ Recent Game Projections -------- #
        recent_avg = { 
            "recent_projection": {
            "rush_attempts": recent_games_rushatt_proj,
            "rush_yards": recent_games_rushyrds_proj,
            "targets": recent_games_targets_proj,
            "rec_yards": recent_games_recyrds_proj,
            "rush_tds": recent_games_rushtds_proj,
            "rec_tds": recent_games_rectds_proj,
            "weight" : 0.4
            },
        }
    else:
        # ----- Previous Similar Games ----- #
        avgs = { "recent_projection": {
            "rush_attempts": 0,
            "rush_yards": 0,
            "targets": 0,
            "rec_yards": 0,
            "rush_tds": 0,
            "rec_tds": 0,
            "weight" : 0
            },
        }

        # ------ Projection from previous games ----- #
        reg_avgs = { "recent_projection": {
            "rush_attempts": 0,
            "rush_yards": 0,
            "targets": 0,
            "rec_yards": 0,
            "rush_tds": 0,
            "rec_tds": 0,
            "weight" : 0
            },
        }

        # ------ Historical Projections -------- #
        hist_avg = { 
            "recent_projection": {
            "rush_attempts": historical_ruatt_proj,
            "rush_yards": historical_ruyards_proj,
            "targets": historical_targets_proj,
            "rec_yards": historical_recyards_proj,
            "rush_tds": historical_rutds_proj,
            "rec_tds": historical_rectds_proj,
            "weight" : 0.35
            },
        }
    
        # ------ Recent Game Projections -------- #
        recent_avg = { 
            "recent_projection": {
            "rush_attempts": recent_games_rushatt_proj,
            "rush_yards": recent_games_rushyrds_proj,
            "targets": recent_games_targets_proj,
            "rec_yards": recent_games_recyrds_proj,
            "rush_tds": recent_games_rushtds_proj,
            "rec_tds": recent_games_rectds_proj,
            "weight" : 0.55
            },
        }

    # -- Calculate regression projections from prev games data -- #

    # Add prev game data to all data
    allData.extend(avgs)
    allData.extend(reg_avgs)
    allData.extend(hist_avg)
    allData.extend(recent_avg)

    #confidence for rec yards
    recent_confidence = mean_confidence_interval(recent_games_recyrds)
    historical_confidence = mean_confidence_interval(historical_recyrds)

    confidence_rec = [(recent_confidence[0]*0.75) + (historical_confidence[0]*0.25), (recent_confidence[1]*0.75) + (historical_confidence[1]*0.25), (recent_confidence[2]*0.75) + (historical_confidence[2]*0.25)]

    print("REC YARD CONF: ", confidence_rec)

    # -- CALCULATE PROJECTION -- #

    #----rush attempts----#
    rush_att_full = 0
    rush_att_full = rush_att_full + ((recent_avg['recent_projection']["rush_attempts"] *recent_avg['recent_projection']["weight"])*rushing_coeff)
    rush_att_full = rush_att_full + ((hist_avg['recent_projection']["rush_attempts"] *hist_avg['recent_projection']["weight"])*rushing_coeff)
    rush_att_full = rush_att_full + ((reg_avgs['recent_projection']["rush_attempts"] *reg_avgs['recent_projection']["weight"])*rushing_coeff)
    rush_att_full = rush_att_full + ((avgs['recent_projection']["rush_attempts"] *avgs['recent_projection']["weight"])*rushing_coeff)
    rush_att_full = rush_att_full + ((win_loss_split['recent_projection']["rush_attempts"] *win_loss_split['recent_projection']["weight"])*rushing_coeff)
    if len(allprevGames)  > 1:
        rush_att_full = rush_att_full + ((all_avgs['recent_projection']["rush_attempts"] *all_avgs['recent_projection']["weight"])*rushing_coeff)
        rush_att_full = rush_att_full + ((all_reg_avgs['recent_projection']["rush_attempts"] *all_reg_avgs['recent_projection']["weight"])*rushing_coeff)

    #----rush tds----#
    rush_tds_full = 0
    rush_tds_full = rush_tds_full + (recent_avg['recent_projection']["rush_tds"] *recent_avg['recent_projection']["weight"])
    rush_tds_full = rush_tds_full + (hist_avg['recent_projection']["rush_tds"] *hist_avg['recent_projection']["weight"])
    rush_tds_full = rush_tds_full + (reg_avgs['recent_projection']["rush_tds"] *reg_avgs['recent_projection']["weight"])
    rush_tds_full = rush_tds_full + (avgs['recent_projection']["rush_tds"] *avgs['recent_projection']["weight"])
    rush_tds_full = rush_tds_full + ((win_loss_split['recent_projection']["rush_tds"] *win_loss_split['recent_projection']["weight"]))
    if len(allprevGames)  > 1:
        rush_tds_full = rush_tds_full + ((all_avgs['recent_projection']["rush_tds"] *all_avgs['recent_projection']["weight"]))
        rush_tds_full = rush_tds_full + ((all_reg_avgs['recent_projection']["rush_tds"] *all_reg_avgs['recent_projection']["weight"]))

    #----rush yds----#
    rush_yds_full = 0
    rush_yds_full = rush_yds_full + ((recent_avg['recent_projection']["rush_yards"] *recent_avg['recent_projection']["weight"]) * rushing_coeff)
    rush_yds_full = rush_yds_full + ((hist_avg['recent_projection']["rush_yards"] *hist_avg['recent_projection']["weight"]) * rushing_coeff)
    rush_yds_full = rush_yds_full + ((reg_avgs['recent_projection']["rush_yards"] *reg_avgs['recent_projection']["weight"]) * rushing_coeff)
    rush_yds_full = rush_yds_full + ((avgs['recent_projection']["rush_yards"] *avgs['recent_projection']["weight"])*rushing_coeff)
    rush_yds_full = rush_yds_full + ((win_loss_split['recent_projection']["rush_yards"] *win_loss_split['recent_projection']["weight"])*rushing_coeff)
    if len(allprevGames) > 1:
        rush_yds_full = rush_yds_full + ((all_avgs['recent_projection']["rush_yards"] *all_avgs['recent_projection']["weight"])*rushing_coeff)
        rush_yds_full = rush_yds_full + ((all_reg_avgs['recent_projection']["rush_yards"] *all_reg_avgs['recent_projection']["weight"])*rushing_coeff)

    #---- targets ----#
    tgt_full = 0
    tgt_full = tgt_full + (recent_avg['recent_projection']["targets"] *recent_avg['recent_projection']["weight"])
    tgt_full = tgt_full + (hist_avg['recent_projection']["targets"] *hist_avg['recent_projection']["weight"])
    tgt_full = tgt_full + (reg_avgs['recent_projection']["targets"] *reg_avgs['recent_projection']["weight"])
    tgt_full = tgt_full + (avgs['recent_projection']["targets"] *avgs['recent_projection']["weight"])
    tgt_full = tgt_full + ((win_loss_split['recent_projection']["targets"] *win_loss_split['recent_projection']["weight"]))
    if len(allprevGames) > 1:
        tgt_full = tgt_full + ((all_avgs['recent_projection']["targets"] *all_avgs['recent_projection']["weight"]))
        tgt_full = tgt_full + ((all_reg_avgs['recent_projection']["targets"] *all_reg_avgs['recent_projection']["weight"]))


    #----rec yds----#
    rec_yds_full = 0
    rec_yds_full = rec_yds_full + (recent_avg['recent_projection']["rec_yards"] *recent_avg['recent_projection']["weight"])
    rec_yds_full = rec_yds_full + (hist_avg['recent_projection']["rec_yards"] *hist_avg['recent_projection']["weight"])
    rec_yds_full = rec_yds_full + (reg_avgs['recent_projection']["rec_yards"] *reg_avgs['recent_projection']["weight"])
    rec_yds_full = rec_yds_full + (avgs['recent_projection']["rec_yards"] *avgs['recent_projection']["weight"])
    rec_yds_full = rec_yds_full + ((win_loss_split['recent_projection']["rec_yards"] *win_loss_split['recent_projection']["weight"]))
    if len(allprevGames) > 1:
        rec_yds_full = rec_yds_full + ((all_avgs['recent_projection']["rec_yards"] *all_avgs['recent_projection']["weight"]))
        rec_yds_full = rec_yds_full + ((all_reg_avgs['recent_projection']["rec_yards"] *all_reg_avgs['recent_projection']["weight"]))

    #----rec tds----#
    rec_tds_full = 0
    rec_tds_full = rec_tds_full + (recent_avg['recent_projection']["rec_tds"] *recent_avg['recent_projection']["weight"])
    rec_tds_full = rec_tds_full + (hist_avg['recent_projection']["rec_tds"] *hist_avg['recent_projection']["weight"])
    rec_tds_full = rec_tds_full + (reg_avgs['recent_projection']["rec_tds"] *reg_avgs['recent_projection']["weight"])
    rec_tds_full = rec_tds_full + (avgs['recent_projection']["rec_tds"] *avgs['recent_projection']["weight"])
    rec_tds_full = rec_tds_full + ((win_loss_split['recent_projection']["rec_tds"] *win_loss_split['recent_projection']["weight"]))
    if len(allprevGames) > 1:
        rec_tds_full = rec_tds_full + ((all_avgs['recent_projection']["rec_tds"] *all_avgs['recent_projection']["weight"]))
        rec_tds_full = rec_tds_full + ((all_reg_avgs['recent_projection']["rec_tds"] *all_reg_avgs['recent_projection']["weight"]))



    print(allData)
    
    rec_yds_full = (rec_yds_full * 0.65) + (precyds.item()*0.35)
    rush_yds_full = (rush_yds_full * 0.65) + (prds.item()*0.35)

    print("REc fore: ",precyds.item())
    print("Rush fore: ",prds.item())


    arr = [rush_att_full, rush_yds_full, rush_tds_full, tgt_full, rec_yds_full, rec_tds_full]

    saveprojection = {
        playerName:{
            'Rush Attempts': rush_att_full,
            'Rush Yards:': rush_yds_full,
            'Rush Tds': rush_tds_full,
            'Targets': tgt_full,
            'Rec Yards': rec_yds_full,
            'Rec Tds' : rec_tds_full
        }
    }
    print(saveprojection)

    with open('data.json') as f:
        data = json.load(f)

    data[playerName] = saveprojection

    with open('data.json', 'w') as f:
        json.dump(data, f, indent=2)

    all_projections.append(saveprojection)
    
    output = pd.DataFrame(np.array([arr]), columns=['Rush Attempts:', 'Rush Yards:', 'Rush Tds:', "Targets", "Recieving Yards", "Rec Tds"])

    #Print prev game data
    if count > 1:
        averages = "Rush Att: "+ str(totalatt) +"\n"+ "Rush Yards: "+ str(totalruyards) +"\n"+"Rush Tds: "+ str(totaltds) +"\n"+"Rec Yards: "+ str(totalrecyards) +"\n"+"Targets: "+ str(totaltarg) +"\n"+"Rec Tds: "+ str(totalrectds) +"\n"

        #Print Regression Projections
        regression = "Rush Att: " +str(attregression) + " Rush Yards: "+ str(yardregression) + " Tds: " + str(rutdregression) + " Targets: "+ str(targregression) + " Rec Yards: "+ str(recyardregression)

        new_rows = [[sg.Text(playerName + " Previous "+ str(count) + " " + loc + " game averages versus the "+ oppTeam)], [sg.Text(averages)],
        [sg.Text("Regression Only Model From Previous Similar Games Projects: ")], [sg.Text(regression)], 
        [sg.Text("Our Model Projects: ")], [sg.Text(output)]
        ]
    else:
        new_rows = [ 
        [sg.Text("Our Model Projects: ")], [sg.Text(output)]
        ]
    window.extend_layout(window, new_rows)
    window.refresh()
    print(output)
    print(confidence)

def runWRProj(playerFirst, playerLast, pposition, oppTeam,oppTeamABR,loc, playerfull):

    prevGames = []
    allprevGames = []
    #Save opponent Win Probability Based on FPI
    oppWinProb = fpi_values[oppTeamABR]
    #init team win 
    teamWinProb = 0
    #init Player Team
    playerTeam = ""
    # ------ Load Player data for Model ----- #
    if playerfull == "":
        playerName = playerFirst + " " + playerLast
    else:
        playerName =playerfull
   # playerName = playerFirst + " " + playerLast
    

    #print(player_home_road)
    # ------ Load Opp Team Data for Model -------- #
    t.home_road(team = oppTeam, season = 2023, avg = True)

    opp_games = tg.get_team_game_log(team = oppTeam, season = 2023)
    #opp_games2 = tg.get_team_game_log(team = oppTeam, season = 2023)
    opp_splits = t.home_road(team = oppTeam, season = 2022, avg = True)

    print(opp_games)
    #Store Opposition Defense statistics 
    opp_rushing_defense = opp_games.loc[:, 'opp_pass_yds'].values
    opp_ru_defense_in_wins = opp_games[(opp_games["result"] == "W")]['opp_pass_yds'].values
    opp_ru_defense_in_loss = opp_games[(opp_games["result"] == "L")]['opp_pass_yds'].values

    rushing_sum = sum(opp_rushing_defense)

    rushing_average = rushing_sum/len(opp_rushing_defense)


    pass_d_w = sum(opp_ru_defense_in_wins)/len(opp_ru_defense_in_wins)
    pass_d_l =sum(opp_ru_defense_in_loss)/len(opp_ru_defense_in_loss)


    yval_wins = range(len(opp_ru_defense_in_wins))
    yval_loss = range(len(opp_ru_defense_in_loss))
    yval = range(len(opp_rushing_defense))
    val = linReg(yval, opp_rushing_defense)[0]

    defensive_ru_allowed = linearRegPredict(val, opp_rushing_defense, yval)[0][0]

    print("Defensive Projection: ", defensive_ru_allowed)
    print("Defensive Average: ", rushing_average)

    #get defensive rushing projection coeff

    rec_coeff = defensive_ru_allowed/rushing_average

    # ---- Get ready to store historical data ---- #
    historical_rec = []
    historical_snap_pct = []
    historical_rushtds = []
    historical_targets = []
    historical_recyrds = []
    historical_rectds = []
    recent_games_rec = []
    recent_games_snap_pct = []
    recent_games_targets = []
    recent_games_recyrds = []
    recent_games_rectds = []

    y2023log = ""
    y2022log = ""

    dfLive = True

    i = 2024
    forecast_pick = -1
    #search through last 4 years data
    while (i >= 2019):
    
        try:
            player_game_log = p.get_player_game_log(player = playerName, position = pposition, season = i)
            dfLive = True
        except:
            print("player didnt return a result")
            dfLive = False


        # ---- Get data from all games for recent games trend ---- #
        if i == 2024 and dfLive == True:
            playerTeam = player_game_log.loc[0]['team']
            recent_games_rec.extend(player_game_log.loc[:,'rec'].values)
            recent_games_snap_pct.extend(player_game_log.loc[:,'snap_pct'].values)
            recent_games_targets.extend(player_game_log.loc[:,'tgt'].values)
            recent_games_recyrds.extend(player_game_log.loc[:,'rec_yds'].values)
            recent_games_rectds.extend(player_game_log.loc[:,'rec_td'].values)
            y2023log = player_game_log
        elif i == 2023 and dfLive == True:
            if playerTeam == "":
                playerTeam = player_game_log.loc[0]['team']
            recent_games_rec.extend(player_game_log.loc[:,'rec'].values)
            recent_games_snap_pct.extend(player_game_log.loc[:,'snap_pct'].values)
            recent_games_targets.extend(player_game_log.loc[:,'tgt'].values)
            recent_games_recyrds.extend(player_game_log.loc[:,'rec_yds'].values)
            recent_games_rectds.extend(player_game_log.loc[:,'rec_td'].values)
            historical_rectds.extend(player_game_log.loc[:,'rec_td'].values)
            historical_recyrds.extend(player_game_log.loc[:,'rec_yds'].values)
            historical_snap_pct.extend(player_game_log.loc[:,'snap_pct'].values)
            historical_rec.extend(player_game_log.loc[:,'rec'].values)
            historical_targets.extend( player_game_log.loc[:,'tgt'].values)
            y2022log = player_game_log
        elif dfLive == True:
            historical_rectds.extend(player_game_log.loc[:,'rec_td'].values)
            historical_recyrds.extend(player_game_log.loc[:,'rec_yds'].values)
            historical_snap_pct.extend(player_game_log.loc[:,'snap_pct'].values)
            historical_rec.extend(player_game_log.loc[:,'rec'].values)
            historical_targets.extend( player_game_log.loc[:,'tgt'].values)
        
        # --------------------------------
        # TODO: Test if this If is working
        # --------------------------------

        isGame = False

        if loc == "Away" and dfLive == True:
        # gets data for previous years similar game if available 
            valRow = player_game_log[(player_game_log["game_location"] == "@") & (player_game_log["opp"] == oppTeamABR)]
            print(valRow)
            if len(valRow.index) > 0:
                rec = valRow.iloc[0]['rec']
                snap_pct = valRow.iloc[0]['snap_pct']
                targets = valRow.iloc[0]['tgt']
                recyards = valRow.iloc[0]['rec_yds']
                rectds = valRow.iloc[0]['rec_td']
                all_rec = valRow.iloc[0]['rec']
                all_snap_pct = valRow.iloc[0]['snap_pct']
                all_targets = valRow.iloc[0]['tgt']
                all_recyards = valRow.iloc[0]['rec_yds']
                all_rectds = valRow.iloc[0]['rec_td']
                isGame = True
        elif loc == "Home" and dfLive == True:
            valRow = player_game_log[(player_game_log["game_location"] != "@") & (player_game_log["opp"] == oppTeamABR)]
            print(valRow)
            if len(valRow.index) > 0:
                rec = valRow.iloc[0]['rec']
                snap_pct = valRow.iloc[0]['snap_pct']
                targets = valRow.iloc[0]['tgt']
                recyards = valRow.iloc[0]['rec_yds']
                rectds = valRow.iloc[0]['rec_td']
                all_rec = valRow.iloc[0]['rec']
                all_snap_pct = valRow.iloc[0]['snap_pct']
                all_targets = valRow.iloc[0]['tgt']
                all_recyards = valRow.iloc[0]['rec_yds']
                all_rectds = valRow.iloc[0]['rec_td']
                isGame = True

        print(isGame)
        # Store game data
        if isGame:
            gamedata = {
                "rec": rec,
                "snap_pct": snap_pct,
                "targets": targets,
                "rec_yards": recyards,
                "rec_tds": rectds,
                "year": i
            }
            prevGames.append(gamedata)
            all_pre_gamedata = {
                "rec": all_rec,
                "snap_pct": all_snap_pct,
                "targets": all_targets,
                "rec_yards": all_recyards,
                "rec_tds": all_rectds,
                "year": i
            }
            allprevGames.append(all_pre_gamedata)
            
        i = i -1

        #END WHILE#
        #---------- CALCULATE HISTORICAL MODEL ----------#

    prec = plotGraph(y2023log, "rec", "WR", y2022log)
    ptds = plotGraph(y2023log, "rec_td", "WR", y2022log)
    pryds = plotGraph(y2023log, "rec_yds", "WR", y2022log)
    ptgt = plotGraph(y2023log, "tgt", "WR", y2022log)

    #snap count
    yvalues = range(len(historical_snap_pct))
    val = linReg(yvalues, historical_snap_pct)[0]
    historical_snap_pct_proj = linearRegPredict(val, historical_snap_pct, yvalues)
    historical_snap_pct_proj = historical_snap_pct_proj[0][0]
    #rec yards
    yvalues = range(len(historical_recyrds))
    val = linReg(yvalues, historical_recyrds)[0]
    historical_recyards_proj = linearRegPredict(val, historical_recyrds, yvalues)
    historical_recyards_proj = historical_recyards_proj[0][0]
    #recTds
    yvalues = range(len(historical_rectds))
    val = linReg(yvalues, historical_rectds)[0]
    historical_rectds_proj = linearRegPredict(val, historical_rectds, yvalues)
    historical_rectds_proj = historical_rectds_proj[0][0]
    #receptions
    yvalues = range(len(historical_rec))
    val = linReg(yvalues, historical_rec)[0]
    historical_rec_proj = linearRegPredict(val, historical_rec, yvalues)
    historical_rec_proj= historical_rec_proj[0][0]
    #targets
    yvalues = range(len(historical_targets))
    val = linReg(yvalues, historical_targets)[0]
    historical_targets_proj = linearRegPredict(val, historical_targets, yvalues)
    historical_targets_proj =historical_targets_proj[0][0]

    # print("historical_ruyards_proj: ", historical_ruyards_proj, " historical_recyards_proj ", historical_recyards_proj, " historical_rectds_proj ", historical_rectds_proj, " historical_rutds_proj ",historical_rutds_proj, " historical_ruatt_proj ", historical_rec_proj , " historical_targets_proj ", historical_targets_proj  )
    #---------- CALCULATE RECENT GAMES MODEL ----------#
    passing_allowed = (pass_d_w * (oppWinProb/100)) + (pass_d_l * (fpi_values[playerTeam]/100))

    pass_val = passing_allowed - rushing_average
    new_val = pass_val/rushing_average 
    print("passing diff: ", pass_val)
    print ("to add to coeff " ,new_val)
    rec_coeff = rec_coeff + new_val

    # 4 year averages 
    recTds_4yr_average = sum(historical_rectds)/len(historical_rectds)
    recyrds_4yr_average = sum(historical_recyrds)/len(historical_recyrds)
    snap_pct_4yr_average = sum(historical_snap_pct)/len(historical_snap_pct)
    rec_4yr_average = sum(historical_rec)/len(historical_rec)
    tar_4yr_average = sum(historical_targets)/len(historical_targets)

    four_year_avgs = { 
        "recent_projection": {
            "rec": rec_4yr_average,
            "snap_pct": snap_pct_4yr_average,
            "targets": tar_4yr_average,
            "rec_yards": recyrds_4yr_average,
            "rec_tds": recTds_4yr_average,
            "weight" : 0.1
        },
    }

    player_home_road = ps.home_road(player = playerName, position = pposition, season = 2022)
    #snap count
    yvalues = range(len(recent_games_snap_pct))
    val = linReg(yvalues, recent_games_snap_pct)[0]
    recent_games_snap_pct_proj = linearRegPredict(val, recent_games_snap_pct, yvalues)
    recent_games_snap_pct_proj = recent_games_snap_pct_proj[0][0]
    #rec yards
    yvalues = range(len(recent_games_recyrds))
    val = linReg(yvalues, recent_games_recyrds)[0]
    recent_games_recyrds_proj = linearRegPredict(val, recent_games_recyrds, yvalues)
    recent_games_recyrds_proj = recent_games_recyrds_proj[0][0]
    #recTds
    yvalues = range(len(recent_games_rectds))
    val = linReg(yvalues, recent_games_rectds)[0]
    recent_games_rectds_proj = linearRegPredict(val, recent_games_rectds, yvalues)
    recent_games_rectds_proj = recent_games_rectds_proj[0][0]
    #rec
    yvalues = range(len(recent_games_rec))
    val = linReg(yvalues, recent_games_rec)[0]
    recent_games_rec_proj = linearRegPredict(val, recent_games_rec, yvalues)
    recent_games_rec_proj= recent_games_rec_proj[0][0]
    #targets
    yvalues = range(len(recent_games_targets))
    val = linReg(yvalues, recent_games_targets)[0]
    recent_games_targets_proj = linearRegPredict(val, recent_games_targets, yvalues)
    recent_games_targets_proj =recent_games_targets_proj[0][0]

    # W/L Splits
    player_w_l_splits = ps.win_loss(player = playerName, position = pposition, season = 2022, avg = True)
    teamWinProb = fpi_values[playerTeam]
    oppTeamWin = fpi_values[oppTeamABR]
    win_loss_split = win_loss_calc (teamWinProb, oppTeamWin, player_w_l_splits, pposition)

    #all games 

    totalsnap_pct = 0
    totaltds = 0
    totalrec = 0
    totaltarg = 0
    totalrecyards = 0
    totalrectds = 0

    count = 0
    yval = []
    snap_pctTrend = []
    recTrend = []
    rutdTrend = []
    rectdTrend = []
    targTrend = []
    recyrdTrend = []

    if len(allprevGames) > 1:

        for game in allprevGames:

            totalsnap_pct = totalsnap_pct + game['snap_pct']
            totalrec = totalrec + game['rec']
            totaltarg = totaltarg + game['targets']
            totalrectds = totalrectds +game['rec_tds']
            totalrecyards = totalrecyards + game['rec_yards']
            count = count + 1
            yval.append(count)
            snap_pctTrend.append (game['snap_pct'])
            recTrend.append (game['rec'])
            rectdTrend.append(game['rec_tds'])
            targTrend.append(game['targets'])
            recyrdTrend.append(game['rec_yards'])
        #rush yards
        val = linReg(yval, snap_pctTrend)[0]
        snap_pctregression = linearRegPredict(val, snap_pctTrend, yval)
        snap_pctregression = snap_pctregression[0][0]
        #rec yards
        val = linReg(yvalues, recyrdTrend)[0]
        recyardregression = linearRegPredict(val, recyrdTrend, yval)
        recyardregression = recyardregression[0][0]
        #recTds
        val = linReg(yvalues, rectdTrend)[0]
        rectdregression = linearRegPredict(val, rectdTrend, yval)
        rectdregression = rectdregression[0][0]
        #rushatt
        val = linReg(yval, recTrend)[0]
        recregression = linearRegPredict(val, recTrend, yval)
        recregression= recregression[0][0]
        #targets
        val = linReg(yvalues, targTrend)[0]
        targregression = linearRegPredict(val, targTrend, yval)
        targregression =targregression[0][0]
        # --- Calc averages from previous 4 games --- #
        totalsnap_pct = totalsnap_pct/count
        totalrec = totalrec/count
        totaltarg = totaltarg/count
        totalrecyards = totalrecyards/count
        totalrectds = totalrectds/count

        all_avgs = { "recent_projection": {
            "rec": totalrec,
            "snap_pct": totalsnap_pct,
            "targets": totaltarg,
            "rec_yards": totalrecyards,
            "rec_tds": totalrectds,
            "weight" : 0.2
            },
        }

        # ------ Projection from previous games ----- #
        allreg_avgs = { "recent_projection": {
            "rec": recregression,
            "snap_pct": snap_pctregression,
            "targets": targregression,
            "rec_yards": recyardregression,
            "rec_tds": rectdregression,
            "weight" : 0.15
            },
        }
    

    totalsnap_pct = 0
    totaltds = 0
    totalrec = 0
    totaltarg = 0
    totalrecyards = 0
    totalrectds = 0

    count = 0
    yval = []
    snap_pctTrend = []
    recTrend = []
    rutdTrend = []
    rectdTrend = []
    targTrend = []
    recyrdTrend = []

    #yards confidence interval
    recent_confidence = mean_confidence_interval(recent_games_recyrds)
    historical_confidence = mean_confidence_interval(historical_recyrds)

    confidence = [(recent_confidence[0]*0.75) + (historical_confidence[0]*0.25), (recent_confidence[1]*0.75) + (historical_confidence[1]*0.25), (recent_confidence[2]*0.75) + (historical_confidence[2]*0.25)]

    #receptions confidence interval
    recent_confidence = mean_confidence_interval(recent_games_rec)
    historical_confidence = mean_confidence_interval(historical_rec)

    rec_confidence = [(recent_confidence[0]*0.75) + (historical_confidence[0]*0.25), (recent_confidence[1]*0.75) + (historical_confidence[1]*0.25), (recent_confidence[2]*0.75) + (historical_confidence[2]*0.25)]

    print("Rec Conf: ", rec_confidence)


    for game in prevGames:

        totalsnap_pct = totalsnap_pct + game['snap_pct']
        totalrec = totalrec + game['rec']
        totaltarg = totaltarg + game['targets']
        totalrectds = totalrectds +game['rec_tds']
        totalrecyards = totalrecyards + game['rec_yards']
        count = count + 1
        yval.append(count)
        snap_pctTrend.append (game['snap_pct'])
        recTrend.append (game['rec'])
        rectdTrend.append(game['rec_tds'])
        targTrend.append(game['targets'])
        recyrdTrend.append(game['rec_yards'])

    # -- Calc Prev game Regression -- #

    # -- Check trend value -- #

    # rutrend = linReg(yval, yardTrend)
    #  rutrend = rutrend[0]

    if count > 1: 
        #rush yards
        val = linReg(yval, snap_pctTrend)[0]
        snap_pctregression = linearRegPredict(val, snap_pctTrend, yval)
        snap_pctregression = snap_pctregression[0][0]
        #rec yards
        val = linReg(yvalues, recyrdTrend)[0]
        recyardregression = linearRegPredict(val, recyrdTrend, yval)
        recyardregression = recyardregression[0][0]
        #recTds
        val = linReg(yvalues, rectdTrend)[0]
        rectdregression = linearRegPredict(val, rectdTrend, yval)
        rectdregression = rectdregression[0][0]
        #rushatt
        val = linReg(yval, recTrend)[0]
        recregression = linearRegPredict(val, recTrend, yval)
        recregression= recregression[0][0]
        #targets
        val = linReg(yvalues, targTrend)[0]
        targregression = linearRegPredict(val, targTrend, yval)
        targregression =targregression[0][0]

    # print(attregression)
    # print (tdRegression)
    if count > 1:
        # --- Calc averages from previous 4 games --- #
        totalsnap_pct = totalsnap_pct/count
        totalrec = totalrec/count
        totaltarg = totaltarg/count
        totalrecyards = totalrecyards/count
        totalrectds = totalrectds/count

        # ----- Previous Similar Games ----- #
        avgs = { "recent_projection": {
            "rec": totalrec,
            "snap_pct": totalsnap_pct,
            "targets": totaltarg,
            "rec_yards": totalrecyards,
            "rec_tds": totalrectds,
            "weight" : 0.1
            },
        }

        # ------ Projection from previous games ----- #
        reg_avgs = { "recent_projection": {
            "rec": recregression,
            "snap_pct": snap_pctregression,
            "targets": targregression,
            "rec_yards": recyardregression,
            "rec_tds": rectdregression,
            "weight" : 0.05
            },
        }

        # ------ Historical Projections -------- #
        hist_avg = { 
            "recent_projection": {
                "rec": historical_rec_proj,
                "snap_pct": historical_snap_pct_proj,
                "targets": historical_targets_proj,
                "rec_yards": historical_recyards_proj,
                "rec_tds": historical_rectds_proj,
                "weight" : 0.1
            },
        }
    
        # ------ Recent Game Projections -------- #
        recent_avg = { 
            "recent_projection": {
                "rec": recent_games_rec_proj,
                "snap_pct": recent_games_snap_pct_proj,
                "targets": recent_games_targets_proj,
                "rec_yards": recent_games_recyrds_proj,
                "rec_tds": recent_games_rectds_proj,
                "weight" : 0.15
            },
        }
    elif len(allprevGames) > 1 and count < 1:
        # ----- Previous Similar Games ----- #
        avgs = { "recent_projection": {
            "rec": 0,
            "snap_pct": 0,
            "targets": 0,
            "rec_yards": 0,
            "rec_tds": 0,
            "weight" : 0
            },
        }

        # ------ Projection from previous games ----- #
        reg_avgs = { "recent_projection": {
            "rec": 0,
            "snap_pct": 0,
            "targets": 0,
            "rec_yards": 0,
            "rec_tds": 0,
            "weight" : 0
            },
        }

        # ------ Historical Projections -------- #
        hist_avg = { 
            "recent_projection": {
                "rec": historical_rec_proj,
                "snap_pct": historical_snap_pct_proj,
                "targets": historical_targets_proj,
                "rec_yards": historical_recyards_proj,
                "rec_tds": historical_rectds_proj,
                "weight" : 0.15
            },
        }
    
        # ------ Recent Game Projections -------- #
        recent_avg = { 
            "recent_projection": {
                "rec": recent_games_rec_proj,
                "snap_pct": recent_games_snap_pct_proj,
                "targets": recent_games_targets_proj,
                "rec_yards": recent_games_recyrds_proj,
                "rec_tds": recent_games_rectds_proj,
                "weight" : 0.25
            },
        }
    else: 
        # ----- Previous Similar Games ----- #
        avgs = { "recent_projection": {
            "rec": 0,
            "snap_pct": 0,
            "targets": 0,
            "rec_yards": 0,
            "rec_tds": 0,
            "weight" : 0
            },
        }

        # ------ Projection from previous games ----- #
        reg_avgs = { "recent_projection": {
            "rec": 0,
            "snap_pct": 0,
            "targets": 0,
            "rec_yards": 0,
            "rec_tds": 0,
            "weight" : 0
            },
        }

        # ------ Historical Projections -------- #
        hist_avg = { 
            "recent_projection": {
                "rec": historical_rec_proj,
                "snap_pct": historical_snap_pct_proj,
                "targets": historical_targets_proj,
                "rec_yards": historical_recyards_proj,
                "rec_tds": historical_rectds_proj,
                "weight" : 0.25
            },
        }
    
        # ------ Recent Game Projections -------- #
        recent_avg = { 
            "recent_projection": {
                "rec": recent_games_rec_proj,
                "snap_pct": recent_games_snap_pct_proj,
                "targets": recent_games_targets_proj,
                "rec_yards": recent_games_recyrds_proj,
                "rec_tds": recent_games_rectds_proj,
                "weight" : 0.5
            },
        }

    # -- Calculate regression projections from prev games data -- #

    # Add prev game data to all data
    allData.extend(avgs)
    allData.extend(reg_avgs)
    allData.extend(hist_avg)
    allData.extend(recent_avg)

    # -- CALCULATE PROJECTION -- #

    #----receptions----#
    rec_full = 0
    rec_full = rec_full + ((recent_avg['recent_projection']["rec"] *recent_avg['recent_projection']["weight"])*rec_coeff)
    rec_full = rec_full + ((hist_avg['recent_projection']["rec"] *hist_avg['recent_projection']["weight"])*rec_coeff)
    rec_full = rec_full + ((reg_avgs['recent_projection']["rec"] *reg_avgs['recent_projection']["weight"])*rec_coeff)
    rec_full = rec_full + ((avgs['recent_projection']["rec"] *avgs['recent_projection']["weight"])*rec_coeff)
    rec_full = rec_full + ((win_loss_split['recent_projection']["rec"] *win_loss_split['recent_projection']["weight"])*rec_coeff)
    rec_full = rec_full + ((four_year_avgs['recent_projection']["rec"] *four_year_avgs['recent_projection']["weight"])*rec_coeff)
    if len(allprevGames) > 1:
        rec_full = rec_full + ((all_avgs['recent_projection']["rec"] *all_avgs['recent_projection']["weight"])*rec_coeff)
        rec_full = rec_full + ((allreg_avgs['recent_projection']["rec"] *allreg_avgs['recent_projection']["weight"])*rec_coeff)


    #----snap count----#
    snap_pct_full = 0
    snap_pct_full = snap_pct_full + ((recent_avg['recent_projection']["snap_pct"] *recent_avg['recent_projection']["weight"])*rec_coeff)
    snap_pct_full = snap_pct_full + ((hist_avg['recent_projection']["snap_pct"] *hist_avg['recent_projection']["weight"])*rec_coeff)
    snap_pct_full = snap_pct_full + ((reg_avgs['recent_projection']["snap_pct"] *reg_avgs['recent_projection']["weight"])*rec_coeff)
    snap_pct_full = snap_pct_full + ((avgs['recent_projection']["snap_pct"] *avgs['recent_projection']["weight"])*rec_coeff)
    snap_pct_full = snap_pct_full + ((win_loss_split['recent_projection']["snap_pct"] *win_loss_split['recent_projection']["weight"])*rec_coeff)
    snap_pct_full = snap_pct_full + ((four_year_avgs['recent_projection']["snap_pct"] *four_year_avgs['recent_projection']["weight"])*rec_coeff)
    if len(allprevGames) > 1:
        snap_pct_full = snap_pct_full + ((all_avgs['recent_projection']["snap_pct"] *all_avgs['recent_projection']["weight"])*rec_coeff)
        snap_pct_full = snap_pct_full + ((allreg_avgs['recent_projection']["snap_pct"] *allreg_avgs['recent_projection']["weight"])*rec_coeff)

    #---- targets ----#
    tgt_full = 0
    tgt_full = tgt_full + ((recent_avg['recent_projection']["targets"] *recent_avg['recent_projection']["weight"])*rec_coeff)
    tgt_full = tgt_full + ((hist_avg['recent_projection']["targets"] *hist_avg['recent_projection']["weight"])*rec_coeff)
    tgt_full = tgt_full + ((reg_avgs['recent_projection']["targets"] *reg_avgs['recent_projection']["weight"])*rec_coeff)
    tgt_full = tgt_full + ((avgs['recent_projection']["targets"] *avgs['recent_projection']["weight"])*rec_coeff)
    tgt_full = tgt_full + ((win_loss_split['recent_projection']["targets"] *win_loss_split['recent_projection']["weight"])*rec_coeff)
    tgt_full = tgt_full + ((four_year_avgs['recent_projection']["targets"] *four_year_avgs['recent_projection']["weight"])*rec_coeff)
    if len(allprevGames) > 1:
        tgt_full = tgt_full + ((all_avgs['recent_projection']["targets"] *all_avgs['recent_projection']["weight"])*rec_coeff)
        tgt_full = tgt_full + ((allreg_avgs['recent_projection']["targets"] *allreg_avgs['recent_projection']["weight"])*rec_coeff)

    #----rec yds----#
    rec_yds_full = 0
    rec_yds_full = rec_yds_full + ((recent_avg['recent_projection']["rec_yards"] *recent_avg['recent_projection']["weight"])*rec_coeff)
    rec_yds_full = rec_yds_full + ((hist_avg['recent_projection']["rec_yards"] *hist_avg['recent_projection']["weight"])*rec_coeff)
    rec_yds_full = rec_yds_full + ((reg_avgs['recent_projection']["rec_yards"] *reg_avgs['recent_projection']["weight"])*rec_coeff)
    rec_yds_full = rec_yds_full + ((avgs['recent_projection']["rec_yards"] *avgs['recent_projection']["weight"])*rec_coeff)
    rec_yds_full = rec_yds_full + ((win_loss_split['recent_projection']["rec_yards"] *win_loss_split['recent_projection']["weight"])*rec_coeff)
    rec_yds_full = rec_yds_full + ((four_year_avgs['recent_projection']["rec_yards"] *four_year_avgs['recent_projection']["weight"])*rec_coeff)
    if len(allprevGames) > 1:
        rec_yds_full = rec_yds_full + ((all_avgs['recent_projection']["rec_yards"] *all_avgs['recent_projection']["weight"])*rec_coeff)
        rec_yds_full = rec_yds_full + ((allreg_avgs['recent_projection']["rec_yards"] *allreg_avgs['recent_projection']["weight"])*rec_coeff)

    #----rec tds----#
    rec_tds_full = 0
    rec_tds_full = rec_tds_full + (recent_avg['recent_projection']["rec_tds"] *recent_avg['recent_projection']["weight"])
    rec_tds_full = rec_tds_full + (hist_avg['recent_projection']["rec_tds"] *hist_avg['recent_projection']["weight"])
    rec_tds_full = rec_tds_full + (reg_avgs['recent_projection']["rec_tds"] *reg_avgs['recent_projection']["weight"])
    rec_tds_full = rec_tds_full + (avgs['recent_projection']["rec_tds"] *avgs['recent_projection']["weight"])
    rec_tds_full = rec_tds_full + ((win_loss_split['recent_projection']["rec_tds"] *win_loss_split['recent_projection']["weight"]))
    rec_tds_full = rec_tds_full + ((four_year_avgs['recent_projection']["rec_tds"] *four_year_avgs['recent_projection']["weight"]))
    if len(allprevGames) > 1:
        rec_tds_full = rec_tds_full + ((all_avgs['recent_projection']["rec_tds"] *all_avgs['recent_projection']["weight"])*rec_coeff)
        rec_tds_full = rec_tds_full + ((allreg_avgs['recent_projection']["rec_tds"] *allreg_avgs['recent_projection']["weight"])*rec_coeff)


    print(rec_coeff)
    print(allData)

    print("Forecast rec yards: ", pryds)
    print("Forecast tds: ", ptds)
    print("Forecast rec: ", prec)
    print("Forecast tgt: ", ptgt)

    print("forecast: ", type(pryds))

    rec_tds_full = (rec_tds_full *0.8) + (ptds.item() * 0.2)
    rec_yds_full = (rec_yds_full *0.8) + (pryds.item() * 0.2)
    tgt_full = (tgt_full*0.8) + (ptgt.item() *0.2)
    rec_full = (rec_full*0.8) + (prec.item() *0.2)

    if forecast_pick > 0:
        rec_yds_full = (rec_yds_full * 0.8) + (forecast_pick[0] *0.2)

    arr = [rec_full, snap_pct_full, tgt_full, rec_yds_full, rec_tds_full]

    saveprojection = {
        playerName: {
            "Receptions": rec_full,
            "Snap Count": snap_pct_full,
            'Targets': tgt_full,
            'Rec Yards': rec_yds_full,
            'Rec Tds': rec_tds_full,
            'Conf Interval': confidence
        }
    }

    all_projections.append(saveprojection)
    
    output = pd.DataFrame(np.array([arr]), columns=['Receptions: ', 'Snap Count:',"Targets", "Recieving Yards", "Rec Tds"])


    with open('data.json') as f:
        data = json.load(f)

    data[playerName] = saveprojection

    with open('data.json', 'w') as f:
        json.dump(data, f)  

    #Print prev game data
    if count > 1:
        averages = "Snap Count: "+ str(totalsnap_pct) +"\n"+ "receptions: "+ str(totalrec)+"\n"+"Rec Yards: "+ str(totalrecyards) +"\n"+"Targets: "+ str(totaltarg) +"\n"+"Rec Tds: "+ str(totalrectds) +"\n"

        #Print Regression Projections
        regression = "Snap Count: " +str(snap_pctregression) + " Receptions: "+ str(recregression) + " Tds: " + str(rectdregression) + " Targets: "+ str(targregression) + " Rec Yards: "+ str(recyardregression)

        new_rows = [[sg.Text(playerName + " Previous "+ str(count) + " " + loc + " game averages versus the "+ oppTeam)], [sg.Text(averages)],
        [sg.Text("Regression Only Model From Previous Similar Games Projects: ")], [sg.Text(regression)], 
        [sg.Text("Our Model Projects: ")], [sg.Text(output)]
        ]
    else:
        new_rows = [ 
        [sg.Text("Our Model Projects: ")], [sg.Text(output)]
        ]
    window.extend_layout(window, new_rows)
    window.refresh()
    print(output)
    print(confidence)

def runQBProj(playerFirst, playerLast, pposition, oppTeam,oppTeamABR,loc, playerfull):
    prevGames = []
    allprevGames = []
    #Save opponent Win Probability Based on FPI
    oppWinProb = fpi_values[oppTeamABR]
    #init team win 
    teamWinProb = 0
    #init Player Team
    playerTeam = ""
   # playerName = playerFirst + " " + playerLast
    if playerfull == "":
        playerName = playerFirst + " " + playerLast
    else:
        playerName =playerfull

    # ------ Load Opp Team Data for Model -------- #

    tg.get_team_game_log(team = oppTeam, season = 2023)
    t.home_road(team = oppTeam, season = 2023, avg = True)

    opp_games = tg.get_team_game_log(team = oppTeam, season = 2023)
    opp_splits = t.home_road(team = oppTeam, season = 2022, avg = True)

    #Store Opposition Defense statistics 
    opp_rushing_defense = opp_games.loc[:, 'opp_rush_yds'].values
    opp_ru_defense_in_wins = opp_games[(opp_games["result"] == "W")]['opp_rush_yds'].values
    opp_ru_defense_in_loss = opp_games[(opp_games["result"] == "L")]['opp_rush_yds'].values

    #Opponent passing defense history
    opp_pass_defense = opp_games.loc[:, 'opp_pass_yds'].values
    opp_pass_defense_in_wins = opp_games[(opp_games["result"] == "W")]['opp_pass_yds'].values
    opp_pass_defense_in_loss = opp_games[(opp_games["result"] == "L")]['opp_pass_yds'].values
    passing_sum = sum(opp_pass_defense)

    pass_d_w = sum(opp_pass_defense_in_wins)/len(opp_pass_defense_in_wins)
    pass_d_l =sum(opp_pass_defense_in_loss)/len(opp_pass_defense_in_loss)

    passing_average = passing_sum/len(opp_pass_defense)

    print("Defense Pass yards allowed Average", passing_average)
    #Win/Loss Splits
    yval_wins = range(len(opp_pass_defense_in_wins))
    yval_loss = range(len(opp_pass_defense_in_loss))

   

    yval = range(len(opp_pass_defense))
    val = linReg(yval, opp_pass_defense)[0]


    defensive_pass_allowed = linearRegPredict(val, opp_pass_defense, yval)[0][0]

    #get defensive passing projection coeff
    print("DPA " , defensive_pass_allowed)
    pass_coeff = defensive_pass_allowed/passing_average
    print("Passing Coeff " , pass_coeff)
    yval = range(len(opp_rushing_defense))
    val = linReg(yval, opp_rushing_defense)[0]
    rushing_sum = sum(opp_rushing_defense)

    rushing_average = rushing_sum/len(opp_rushing_defense)

    defensive_ru_allowed = linearRegPredict(val, opp_rushing_defense, yval)[0][0]

    #get defensive rushing projection coeff

    rushing_coeff = defensive_ru_allowed/rushing_average

    # ---- Get ready to store historical data ---- #

    log2023 = None
    log2022 = None
    historical_rushatt = []
    historical_rushyrds = []
    historical_rushtds = []
    historical_cmp = []
    historical_att = []
    historical_pass_td = []
    historical_int = []
    historical_rating = []
    historical_pass_yds = []

    recent_games_rushatt = []
    recent_games_rushyrds = []
    recent_games_rushtds = []
    recent_games_cmp = []
    recent_games_att = []
    recent_games_pass_td = []
    recent_games_int = []
    recent_games_rating = []
    recent_games_pass_yds = []
    dfLive = True
    playerTeam = ""

    i = 2024
    forecast_pick = -1

    #search through last 4 years data
    while (i >= 2019):
    
        try:
            player_game_log = p.get_player_game_log(player = playerName, position = pposition, season = i)
            dfLive = True
        except:
            print("player didnt return a result")
            dfLive = False

        # ---- Get data from all games for recent games trend ---- #
        if i == 2024 and dfLive:
            playerTeam = player_game_log.loc[0]['team']
            y2023_rushatt = player_game_log.loc[:,'rush_att'].values
            y2023_rushyrds = player_game_log.loc[:,'rush_yds'].values
            y2023_rushtds = player_game_log.loc[:,'rush_td'].values
            y2023_cmp = player_game_log.loc[:,'cmp'].values
            y2023_att = player_game_log.loc[:,'att'].values
            y2023_pass_td = player_game_log.loc[:,'pass_td'].values
            y2023_int = player_game_log.loc[:,'int'].values
            y2023_rating = player_game_log.loc[:,'rating'].values
            y2023_pass_yds = player_game_log.loc[:,'pass_yds'].values
            log2023 = player_game_log
            print(player_game_log)

        elif i == 2023 and dfLive:
            recent_games_rushatt.extend(player_game_log.loc[:,'rush_att'].values)
            recent_games_rushyrds.extend(player_game_log.loc[:,'rush_yds'].values)
            recent_games_rushtds.extend(player_game_log.loc[:,'rush_td'].values)
            recent_games_cmp.extend(player_game_log.loc[:,'cmp'].values)
            recent_games_att.extend(player_game_log.loc[:,'att'].values)
            recent_games_pass_td.extend(player_game_log.loc[:,'pass_td'].values)
            recent_games_int.extend(player_game_log.loc[:,'int'].values)
            recent_games_rating.extend(player_game_log.loc[:,'rating'].values)
            recent_games_pass_yds.extend(player_game_log.loc[:, 'pass_yds'].values)
            log2022 = player_game_log
            print(player_game_log)
            if playerTeam == "":
                playerTeam = player_game_log.loc[0]['team']
            
            historical_rushyrds.extend(player_game_log.loc[:,'rush_yds'].values)
            historical_int.extend(player_game_log.loc[:,'int'].values)
            historical_cmp.extend(player_game_log.loc[:,'cmp'].values)
            historical_att.extend(player_game_log.loc[:,'att'].values)
            historical_rushtds.extend(player_game_log.loc[:,'rush_td'].values)
            historical_rushatt.extend(player_game_log.loc[:,'rush_att'].values)
            historical_pass_td.extend( player_game_log.loc[:,'pass_td'].values)
            historical_pass_yds.extend( player_game_log.loc[:,'pass_yds'].values)
            historical_rating.extend( player_game_log.loc[:,'rating'].values)
            print(player_game_log)
        elif dfLive:
            historical_rushyrds.extend(player_game_log.loc[:,'rush_yds'].values)
            historical_int.extend(player_game_log.loc[:,'int'].values)
            historical_cmp.extend(player_game_log.loc[:,'cmp'].values)
            historical_att.extend(player_game_log.loc[:,'att'].values)
            historical_rushtds.extend(player_game_log.loc[:,'rush_td'].values)
            historical_rushatt.extend(player_game_log.loc[:,'rush_att'].values)
            historical_pass_td.extend( player_game_log.loc[:,'pass_td'].values)
            historical_pass_yds.extend( player_game_log.loc[:,'pass_yds'].values)
            historical_rating.extend( player_game_log.loc[:,'rating'].values)
            print(player_game_log)

        isGame = False

        if loc == "Away" and i >= 2023:
        # gets data for previous years similar game if available 
            valRow = player_game_log[(player_game_log["game_location"] == "@") & (player_game_log["opp"] == oppTeamABR)]
            print(valRow)
            if len(valRow.index) > 0:
                ruattemps = valRow.iloc[0]['rush_att']
                ruyards = valRow.iloc[0]['rush_yds']
                rutds = valRow.iloc[0]['rush_td']
                ints = valRow.iloc[0]['int']
                cmp = valRow.iloc[0]['cmp']
                pass_yds = valRow.iloc[0]['pass_yds']
                pass_tds =  valRow.iloc[0]['pass_td']
                att = valRow.iloc[0]['att']
                rating = valRow.iloc[0]['rating']
                isGame = True
        elif loc == "Home" and i >= 2023:
            valRow = player_game_log[(player_game_log["game_location"] != "@") & (player_game_log["opp"] == oppTeamABR)]
            print(valRow)
            if len(valRow.index) > 0:
                ruattemps = valRow.iloc[0]['rush_att']
                ruyards = valRow.iloc[0]['rush_yds']
                rutds = valRow.iloc[0]['rush_td']
                ints = valRow.iloc[0]['int']
                cmp = valRow.iloc[0]['cmp']
                pass_yds = valRow.iloc[0]['pass_yds']
                pass_tds =  valRow.iloc[0]['pass_td']
                att = valRow.iloc[0]['att']
                rating = valRow.iloc[0]['rating']
                isGame = True

        print(isGame)
        # Store game data
        if isGame:
            gamedata = {
                "rush_attempts": ruattemps,
                "rush_yards": ruyards,
                "rush_tds": rutds,
                "att": att,
                "cmp": cmp,
                "pass_tds": pass_tds,
                "pass_yds": pass_yds,
                "rating": rating,
                "int": ints,
                "year": i
            }
            prevGames.append(gamedata)
        i = i - 1
        #END WHILE#

    #---------- CALCULATE HISTORICAL MODEL ----------#
    passing_allowed = (pass_d_w * (oppWinProb/100)) + (pass_d_l * (fpi_values[playerTeam]/100))
  #  pass_coeff = passing_allowed/passing_average
  #  print(pass_coeff)

    print("Wins: ", pass_d_w)
    print("Loss: ", pass_d_l)
    print("passing avg: ", passing_average)
    print ("Passing allowed average " ,passing_allowed)

    # CALC 2023 data
    pyds = plotGraph(log2023, "pass_yds", "QB", log2022)
    ptds = plotGraph(log2023, "pass_td", "QB", log2022)
    pints = plotGraph(log2023, "int", "QB", log2022)
    prds = plotGraph(log2023, "rush_yds", "QB", log2022)

    y2023_rushyrds = sum(y2023_rushyrds)/len(y2023_rushyrds)
    y2023_pass_td = sum(y2023_pass_td)/len(y2023_pass_td)
    y2023_pass_yds = sum(y2023_pass_yds)/len(y2023_pass_yds)

    #y2023_rushatt
   # y2023_rushyrds
   # y2023_rushtds 
   # y2023_cmp 
   # y2023_att 
   # y2023_pass_td
   # y2023_int 
   # y2023_rating 
   # y2023_pass_yds

    pass_val = passing_allowed - passing_average
    new_val = pass_val/passing_average 
    print("passing diff: ", pass_val)
    print ("to add to coeff " ,new_val)
    pass_coeff = pass_coeff + new_val
    print("Coeff ", pass_coeff)

    #Calc Confidence intervals
    recent_confidence = mean_confidence_interval(recent_games_pass_yds)
    historical_confidence = mean_confidence_interval(historical_pass_yds)

    confidence = [(recent_confidence[0]*0.75) + (historical_confidence[0]*0.25), (recent_confidence[1]*0.75) + (historical_confidence[1]*0.25), (recent_confidence[2]*0.75) + (historical_confidence[2]*0.25)]

    #pass Tds
    recent_confidence = mean_confidence_interval(recent_games_pass_td)
    historical_confidence = mean_confidence_interval(historical_pass_td)

    td_confidence = [(recent_confidence[0]*0.75) + (historical_confidence[0]*0.25), (recent_confidence[1]*0.75) + (historical_confidence[1]*0.25), (recent_confidence[2]*0.75) + (historical_confidence[2]*0.25)]

    #rush yards
    recent_confidence = mean_confidence_interval(recent_games_rushyrds)
    historical_confidence = mean_confidence_interval(historical_rushyrds)

    rush_confidence = [(recent_confidence[0]*0.75) + (historical_confidence[0]*0.25), (recent_confidence[1]*0.75) + (historical_confidence[1]*0.25), (recent_confidence[2]*0.75) + (historical_confidence[2]*0.25)]

    print("Pass TD Conf: ", td_confidence)
    print("Rush Yards Conf: ", rush_confidence)

    #rush yards
    yvalues = range(len(historical_rushyrds))
    val = linReg(yvalues, historical_rushyrds)[0]
    historical_ruyards_proj = linearRegPredict(val, historical_rushyrds, yvalues)
    historical_ruyards_proj = historical_ruyards_proj[0][0]
    #pass yards
    yvalues = range(len(historical_pass_yds))
    val = linReg(yvalues, historical_pass_yds)[0]
    historical_pass_yds_proj = linearRegPredict(val, historical_pass_yds, yvalues)[0][0]
    #passTds
    yvalues = range(len(historical_pass_td))
    val = linReg(yvalues, historical_pass_td)[0]
    historical_passtds_proj = linearRegPredict(val, historical_pass_td, yvalues)[0][0]
    #rushTds
    yvalues = range(len(historical_rushtds))
    val = linReg(yvalues, historical_rushtds)[0]
    historical_rutds_proj = linearRegPredict(val, historical_rushtds, yvalues)
    historical_rutds_proj =historical_rutds_proj[0][0]
    #rushatt
    yvalues = range(len(historical_rushatt))
    val = linReg(yvalues, historical_rushatt)[0]
    historical_ruatt_proj = linearRegPredict(val, historical_rushatt, yvalues)
    historical_ruatt_proj= historical_ruatt_proj[0][0]
    #att
    yvalues = range(len(historical_att))
    val = linReg(yvalues, historical_att)[0]
    historical_att_proj = linearRegPredict(val, historical_att, yvalues)[0][0]
    #cmp
    yvalues = range(len(historical_cmp))
    val = linReg(yvalues, historical_cmp)[0]
    historical_cmp_proj = linearRegPredict(val, historical_cmp, yvalues)[0][0]
    #int
    yvalues = range(len(historical_int))
    val = linReg(yvalues, historical_int)[0]
    historical_int_proj = linearRegPredict(val, historical_int, yvalues)[0][0]
    #rating
    yvalues = range(len(historical_rating))
    val = linReg(yvalues, historical_rating)[0]
    historical_rating_proj = linearRegPredict(val, historical_rating, yvalues)[0][0]

    #---------- CALCULATE RECENT GAMES MODEL ----------#

    #rush yards
    yvalues = range(len(recent_games_rushyrds))
    val = linReg(yvalues, recent_games_rushyrds)[0]
    recent_games_rushyrds_proj = linearRegPredict(val, recent_games_rushyrds, yvalues)
    recent_games_rushyrds_proj = recent_games_rushyrds_proj[0][0]
    #pass yards
    yvalues = range(len(recent_games_pass_yds))
    val = linReg(yvalues, recent_games_pass_yds)[0]
    recent_games_passyrds_proj = linearRegPredict(val, recent_games_pass_yds, yvalues)[0][0]
    #passTds
    yvalues = range(len(recent_games_pass_td))
    val = linReg(yvalues, recent_games_pass_td)[0]
    recent_games_passtds_proj = linearRegPredict(val, recent_games_pass_td, yvalues)[0][0]
    #rushTds
    yvalues = range(len(recent_games_rushtds))
    val = linReg(yvalues, recent_games_rushtds)[0]
    recent_games_rushtds_proj = linearRegPredict(val, recent_games_rushtds, yvalues)
    recent_games_rushtds_proj =recent_games_rushtds_proj[0][0]
    #rushatt
    yvalues = range(len(recent_games_rushatt))
    val = linReg(yvalues, recent_games_rushatt)[0]
    recent_games_rushatt_proj = linearRegPredict(val, recent_games_rushatt, yvalues)
    recent_games_rushatt_proj= recent_games_rushatt_proj[0][0]
    #att
    yvalues = range(len(recent_games_att))
    val = linReg(yvalues, recent_games_att)[0]
    recent_games_att_proj = linearRegPredict(val, recent_games_att, yvalues)[0][0]
    #cmp
    yvalues = range(len(recent_games_cmp))
    val = linReg(yvalues, recent_games_cmp)[0]
    recent_games_cmp_proj = linearRegPredict(val, recent_games_cmp, yvalues)[0][0]
    #int
    yvalues = range(len(recent_games_int))
    val = linReg(yvalues, recent_games_int)[0]
    recent_games_int_proj = linearRegPredict(val, recent_games_int, yvalues)[0][0]
    #rating
    yvalues = range(len(recent_games_rating))
    val = linReg(yvalues, recent_games_rating)[0]
    recent_games_rating_proj = linearRegPredict(val, recent_games_rating, yvalues)[0][0]

    # -------- Previous Games ---------  #

    totalruyards = 0
    totaltds = 0
    totalruatt = 0
    totalatt = 0
    totalcmp = 0
    totalpassyards = 0
    totalpasstds = 0
    totalints = 0
    totalrating = 0

    count = 0
    yval = []
    yardTrend = []
    ruattTrend = []
    rutdTrend = []
    passydsTrend = []
    passtdTrend = []
    attTrend = []
    cmpTrend = []
    intTrend = []
    ratingTrend =[]

    for game in prevGames:

        totalruyards = totalruyards + game['rush_yards']
        totaltds = totaltds + game['rush_tds']
        totalruatt = totalruatt + game['rush_attempts']
        totalatt = totalatt + game['att']
        totalpassyards = totalpassyards +game['pass_yds']
        totalpasstds = totalpasstds + game['pass_tds']
        totalcmp = totalcmp +game['cmp']
        totalints = totalints + game['int']
        totalrating = totalrating + game['rating']

        count = count + 1

        yval.append(count)
        yardTrend.append (game['rush_yards'])
        ruattTrend.append (game['rush_attempts'])
        rutdTrend.append(game['rush_tds'])
        passtdTrend.append(game['pass_tds'])
        attTrend.append(game['att'])
        passydsTrend.append(game['pass_yds'])
        cmpTrend.append(game['cmp'])
        intTrend.append(game['int'])
        ratingTrend.append(game['rating'])
    
    if count > 1: 
        #rush yards
        val = linReg(yval, yardTrend)[0]
        yardregression = linearRegPredict(val, yardTrend, yval)
        yardregression = yardregression[0][0]
        #pass yards
        val = linReg(yvalues, passydsTrend)[0]
        passyardregression = linearRegPredict(val, passydsTrend, yval)
        passyardregression = passyardregression[0][0]
        #passTds
        val = linReg(yvalues, passtdTrend)[0]
        passtdregression = linearRegPredict(val, passtdTrend, yval)
        passtdregression = passtdregression[0][0]
        #rushTds
        val = linReg(yvalues, rutdTrend)[0]
        rutdregression = linearRegPredict(val, rutdTrend, yval)
        rutdregression =rutdregression[0][0]
        #rushatt
        val = linReg(yval, ruattTrend)[0]
        ruattregression = linearRegPredict(val, ruattTrend, yval)
        ruattregression= ruattregression[0][0]
        #att
        val = linReg(yvalues, attTrend)[0]
        attregression = linearRegPredict(val, attTrend, yval)
        attregression =attregression[0][0]
        #cmp
        val = linReg(yvalues, cmpTrend)[0]
        cmpregression = linearRegPredict(val, cmpTrend, yval)
        cmpregression =cmpregression[0][0]
        #int
        val = linReg(yvalues, intTrend)[0]
        intregression = linearRegPredict(val, intTrend, yval)
        intregression =intregression[0][0]
        #rating
        val = linReg(yvalues, ratingTrend)[0]
        ratingregression = linearRegPredict(val, ratingTrend, yval)
        ratingregression =ratingregression[0][0]


        # print(attregression)
        # print (tdRegression)

        # --- Calc averages from previous 4 games --- #
        totalruyards = totalruyards/count
        totaltds = totaltds/count
        totalatt = totalatt/count
        totalruatt = totalruatt/count
        totalcmp = totalcmp/count
        totalints = totalints/count
        totalrating = totalrating/count
        totalpassyards = totalpassyards/count
        totalpasstds = totalpasstds/count
    
    player_w_l_splits = ps.win_loss(player = playerName, position = pposition, season = 2022, avg = True)
    teamWinProb = fpi_values[playerTeam]
    oppTeamWin = fpi_values[oppTeamABR]
    win_loss_split = win_loss_calc (teamWinProb, oppTeamWin, player_w_l_splits, pposition)

    if count > 1:
        # ----- Previous Similar Games ----- #
        avgs = { "recent_projection": {
            "rush_attempts": totalruatt,
            "rush_yards": totalruyards,
            "att": totalatt,
            "pass_yards": totalpassyards,
            "rush_tds": totaltds,
            "pass_tds": totalpasstds,
            "pass_yds": totalpassyards,
            "int": totalints,
            "rating": totalrating,
            "cmp": totalcmp,
            "weight" : 0.15
            },
        }

        # ------ Projection from previous games ----- #
        reg_avgs = { "recent_projection": {
            "rush_attempts": ruattregression,
            "rush_yards": yardregression,
            "att": attregression,
            "pass_yards": passyardregression,
            "rush_tds": rutdregression,
            "pass_tds": passtdregression,
            "int": intregression,
            "rating": ratingregression,
            "cmp": cmpregression,
            "weight" : 0.15
            },
        }

        # ------ Historical Projections -------- #
        hist_avg = { 
            "recent_projection": {
            "rush_attempts": historical_ruatt_proj,
            "rush_yards": historical_ruyards_proj,
            "att": historical_att_proj,
            "pass_yards": historical_pass_yds_proj,
            "rush_tds": historical_rutds_proj,
            "pass_tds": historical_passtds_proj,
            "cmp": historical_cmp_proj,
            "int": historical_int_proj,
            "rating": historical_rating_proj,
            "weight" : 0.35
            },
        }
    
        # ------ Recent Game Projections -------- #
        recent_avg = { 
            "recent_projection": {
            "rush_attempts": recent_games_rushatt_proj,
            "rush_yards": recent_games_rushyrds_proj,
            "att": recent_games_att_proj,
            "pass_yards": recent_games_passyrds_proj,
            "rush_tds": recent_games_rushtds_proj,
            "pass_tds": recent_games_passtds_proj,
            "cmp": recent_games_cmp_proj,
            "int": recent_games_int_proj,
            "rating": recent_games_rating_proj,
            "weight" : 0.35
            },
        }
    else:
        # ----- Previous Similar Games ----- #
        avgs = { "recent_projection": {
            "rush_attempts": 0,
            "rush_yards": 0,
            "att": 0,
            "cmp": 0,
            "pass_yards": 0,
            "rush_tds": 0,
            "pass_tds": 0,
            "pass_yds": 0,
            "int": 0,
            "rating": 0,
            "weight" : 0
            },
        }

        # ------ Projection from previous games ----- #
        reg_avgs = { "recent_projection": {
            "rush_attempts": 0,
            "rush_yards": 0,
            "att": 0,
            "pass_yards": 0,
            "rush_tds": 0,
            "pass_tds": 0,
            "int": 0,
            "rating": 0,
            "cmp": 0,
            "weight" : 0
            },
        }

        # ------ Historical Projections -------- #
        hist_avg = { 
            "recent_projection": {
            "rush_attempts": historical_ruatt_proj,
            "rush_yards": historical_ruyards_proj,
            "att": historical_att_proj,
            "pass_yards": historical_pass_yds_proj,
            "rush_tds": historical_rutds_proj,
            "pass_tds": historical_passtds_proj,
            "cmp": historical_cmp_proj,
            "int": historical_int_proj,
            "rating": historical_rating_proj,
            "weight" : 0.4
            },
        }
    
        # ------ Recent Game Projections -------- #
        recent_avg = { 
            "recent_projection": {
            "rush_attempts": recent_games_rushatt_proj,
            "rush_yards": recent_games_rushyrds_proj,
            "att": recent_games_att_proj,
            "pass_yards": recent_games_passyrds_proj,
            "rush_tds": recent_games_rushtds_proj,
            "pass_tds": recent_games_passtds_proj,
            "cmp": recent_games_cmp_proj,
            "int": recent_games_int_proj,
            "rating": recent_games_rating_proj,
            "weight" : 0.6
            },
        }
    # ENDIF #
    # ---- CALC ---- #
    #----rush attempts----#

    rush_att_full = 0
    rush_att_full = rush_att_full + ((recent_avg['recent_projection']["rush_attempts"] *recent_avg['recent_projection']["weight"])*rushing_coeff)
    rush_att_full = rush_att_full + ((hist_avg['recent_projection']["rush_attempts"] *hist_avg['recent_projection']["weight"])*rushing_coeff)
    rush_att_full = rush_att_full + ((reg_avgs['recent_projection']["rush_attempts"] *reg_avgs['recent_projection']["weight"])*rushing_coeff)
    rush_att_full = rush_att_full + ((avgs['recent_projection']["rush_attempts"] *avgs['recent_projection']["weight"])*rushing_coeff)
    rush_att_full = rush_att_full + ((win_loss_split['recent_projection']["rush_attempts"] *win_loss_split['recent_projection']["weight"])*rushing_coeff)

    #----rush tds----#
    rush_tds_full = 0
    rush_tds_full = rush_tds_full + (recent_avg['recent_projection']["rush_tds"] *recent_avg['recent_projection']["weight"])
    rush_tds_full = rush_tds_full + (hist_avg['recent_projection']["rush_tds"] *hist_avg['recent_projection']["weight"])
    rush_tds_full = rush_tds_full + (reg_avgs['recent_projection']["rush_tds"] *reg_avgs['recent_projection']["weight"])
    rush_tds_full = rush_tds_full + (avgs['recent_projection']["rush_tds"] *avgs['recent_projection']["weight"])
    rush_tds_full = rush_tds_full + ((win_loss_split['recent_projection']["rush_tds"] *win_loss_split['recent_projection']["weight"])*rushing_coeff)

    #----rush yds----#
    rush_yds_full = 0
    rush_yds_full = rush_yds_full + ((recent_avg['recent_projection']["rush_yards"] *recent_avg['recent_projection']["weight"]) * rushing_coeff)
    rush_yds_full = rush_yds_full + ((hist_avg['recent_projection']["rush_yards"] *hist_avg['recent_projection']["weight"]) * rushing_coeff)
    rush_yds_full = rush_yds_full + ((reg_avgs['recent_projection']["rush_yards"] *reg_avgs['recent_projection']["weight"]) * rushing_coeff)
    rush_yds_full = rush_yds_full + ((avgs['recent_projection']["rush_yards"] *avgs['recent_projection']["weight"])*rushing_coeff)
    rush_yds_full = rush_yds_full + ((win_loss_split['recent_projection']["rush_yards"] *win_loss_split['recent_projection']["weight"])*rushing_coeff)

    #---- completions ----#
    cmp_full = 0
    cmp_full = cmp_full + ((recent_avg['recent_projection']["cmp"] *recent_avg['recent_projection']["weight"])*pass_coeff)
    cmp_full = cmp_full + ((hist_avg['recent_projection']["cmp"] *hist_avg['recent_projection']["weight"])*pass_coeff)
    cmp_full = cmp_full + ((reg_avgs['recent_projection']["cmp"] *reg_avgs['recent_projection']["weight"])*pass_coeff)
    cmp_full = cmp_full + ((avgs['recent_projection']["cmp"] *avgs['recent_projection']["weight"])*pass_coeff)
    cmp_full = cmp_full + ((win_loss_split['recent_projection']["cmp"] *win_loss_split['recent_projection']["weight"])*pass_coeff)

    #---- att ----#
    att_full = 0
    att_full = att_full + ((recent_avg['recent_projection']["att"] *recent_avg['recent_projection']["weight"])*pass_coeff)
    att_full = att_full + ((hist_avg['recent_projection']["att"] *hist_avg['recent_projection']["weight"])*pass_coeff)
    att_full = att_full + ((reg_avgs['recent_projection']["att"] *reg_avgs['recent_projection']["weight"])*pass_coeff)
    att_full = att_full + ((avgs['recent_projection']["att"] *avgs['recent_projection']["weight"])*pass_coeff)

    #---- int ----#
    int_full = 0
    int_full = int_full + ((recent_avg['recent_projection']["int"] *recent_avg['recent_projection']["weight"]))
    int_full = int_full + ((hist_avg['recent_projection']["int"] *hist_avg['recent_projection']["weight"]))
    int_full = int_full + ((reg_avgs['recent_projection']["int"] *reg_avgs['recent_projection']["weight"]))
    int_full = int_full + ((avgs['recent_projection']["int"] *avgs['recent_projection']["weight"]))
    int_full = int_full + ((win_loss_split['recent_projection']["int"] *win_loss_split['recent_projection']["weight"])*pass_coeff)

    #---- rating ----#
    rating_full = 0
    rating_full = rating_full + ((recent_avg['recent_projection']["rating"] *recent_avg['recent_projection']["weight"])*pass_coeff)
    rating_full = rating_full + ((hist_avg['recent_projection']["rating"] *hist_avg['recent_projection']["weight"])*pass_coeff)
    rating_full = rating_full + ((reg_avgs['recent_projection']["rating"] *reg_avgs['recent_projection']["weight"])*pass_coeff)
    rating_full = rating_full + ((avgs['recent_projection']["rating"] *avgs['recent_projection']["weight"])*pass_coeff)
    
    #----pass yds----#
    pass_yds_full = 0
    pass_yds_full = pass_yds_full + ((recent_avg['recent_projection']["pass_yards"] *recent_avg['recent_projection']["weight"]) *pass_coeff)
    pass_yds_full = pass_yds_full + ((hist_avg['recent_projection']["pass_yards"] *hist_avg['recent_projection']["weight"])*pass_coeff)
    pass_yds_full = pass_yds_full + ((reg_avgs['recent_projection']["pass_yards"] *reg_avgs['recent_projection']["weight"])*pass_coeff)
    pass_yds_full = pass_yds_full + ((avgs['recent_projection']["pass_yards"] *avgs['recent_projection']["weight"])*pass_coeff)
    pass_yds_full = pass_yds_full + ((win_loss_split['recent_projection']["pass_yards"] *win_loss_split['recent_projection']["weight"])*pass_coeff)
    

    #----pass tds----#
    pass_tds_full = 0
    pass_tds_full = pass_tds_full + (recent_avg['recent_projection']["pass_tds"] *recent_avg['recent_projection']["weight"])
    pass_tds_full = pass_tds_full + (hist_avg['recent_projection']["pass_tds"] *hist_avg['recent_projection']["weight"])
    pass_tds_full = pass_tds_full + (reg_avgs['recent_projection']["pass_tds"] *reg_avgs['recent_projection']["weight"])
    pass_tds_full = pass_tds_full + (avgs['recent_projection']["pass_tds"] *avgs['recent_projection']["weight"])
    pass_tds_full = pass_tds_full + ((win_loss_split['recent_projection']["pass_tds"] *win_loss_split['recent_projection']["weight"])*pass_coeff)



    print(allData)

    print("Forecast pass yards: ", pyds)
    print("Forecast pass tds: ", ptds)
    print("Forecast rush yards: ", prds)
    print("Forecast ints: ", pints)

    print("forecast: ",pints.item() )
    print("forecast: ", type(pints))

    pass_yds_full = (pass_yds_full *0.5) + (y2023_pass_yds * 0.1) + (pyds.item() * 0.4)
    pass_tds_full = (pass_tds_full *0.5) + (y2023_pass_td * 0.1) + (ptds.item() * 0.4)
    rush_yds_full = (rush_yds_full*0.5) + (y2023_rushyrds * 0.1) + (prds.item() *0.4)

     

    arr = [rush_att_full, rush_yds_full, rush_tds_full, att_full, cmp_full, int_full, pass_yds_full, pass_tds_full, rating_full ]
    
    output = pd.DataFrame(np.array([arr]), columns=['Rush Attempts:', 'Rush Yards:', 'Rush Tds:', "Attemps", "Completions", "ints", "Pass Yards", "Pass Tds", "rating"])
    print(output)

    print("confidence for yards: ", confidence)

    saveprojection = {
        playerName: {
            "Rush Attempts": rush_att_full,
            "Rush Yards": rush_yds_full,
            "Rush Tds": rush_tds_full,
            "pAttempts": att_full,
            "completions": cmp_full,
            "interceptions": int_full,
            "pass yards": pass_yds_full,
            "pass tds": pass_tds_full
        }
    }

    print("PROJECTIONS: ", saveprojection)

    all_projections.append(saveprojection)

    #Print prev game data
    if count > 1:
        averages = "Rush Att: "+ str(totalatt) +"\n"+ "Rush Yards: "+ str(totalruyards) +"\n"+"Rush Tds: "+ str(totaltds) +"\n"+"Pass Yards: "+ str(totalpassyards) +"\n"+"att: "+ str(totalatt) +"\n"+"Pass Tds: "+ str(totalpasstds) +"\n"

        #Print Regression Projections
        regression = "Rush Att: " +str(attregression) + " Rush Yards: "+ str(yardregression) + " Tds: " + str(rutdregression) + " ints: "+ str(intregression) + " pass Yards: "+ str(passyardregression)

        new_rows = [[sg.Text(playerName + " Previous "+ str(count) + " " + loc + " game averages versus the "+ oppTeam)], [sg.Text(averages)],
        [sg.Text("Regression Only Model From Previous Similar Games Projects: ")], [sg.Text(regression)], 
        [sg.Text("Our Model Projects: ")], [sg.Text(output)]
        ]
    else:
        new_rows = [ 
        [sg.Text("Our Model Projects: ")], [sg.Text(output)]
        ]
    window.extend_layout(window, new_rows)
    window.refresh()

# ----------- Create the 3 layouts this Window will display -----------
layout1 = [[sg.Text('Model Next Game Stats')],
           [sg.Text("Player First Name")],
           [sg.Input(key='-IN-')],
           [sg.Text("Player Last Name")],
           [sg.Input(key='-IN2-')],
           [sg.Text("Player Position")],
           [sg.Input(key='-IN3-')],
           [sg.Text("Opposing Team Next Game")],
           [sg.Input(key='-IN4-')],
           [sg.Text("Home Or Away Next Game")],
           [sg.Input(key='-IN5-')],
           [sg.Text("Opposing Team Abreviation")],
           [sg.Input(key='-IN6-')],
           [sg.Button('Submit')]]

layout2 = [[sg.Text('This is layout 1 - It is all Checkboxes')],
           *[[sg.CB(f'Checkbox {i}')] for i in range(5)]]

layout3 = [[sg.Text('This is layout 3 - It is all Radio Buttons')],
           *[[sg.R(f'Radio {i}', 1)] for i in range(8)]]

# ----------- Create actual layout using Columns and a row of Buttons
layout = [[sg.Column(layout1, key='-COL1-'), sg.Column(layout2, visible=False, key='-COL2-'), sg.Column(layout3, visible=False, key='-COL3-')],
          [sg.Button('Cycle Layout'), sg.Button('1'), sg.Button('2'), sg.Button('3'), sg.Button('Exit')]]

window = sg.Window('Player Projection Model', layout, resizable=True)

layout = 1  # The currently visible layout
while True:
    event, values = window.read()
    print(event, values)
    if event in (None, 'Exit'):
        break
    if event == "Submit":
        playerFirst = values['-IN-']
        playerLast = values['-IN2-']
        pposition = values['-IN3-']
        oppTeam = values['-IN4-']
        oppTeamABR = values['-IN6-']
        loc = values['-IN5-']
        prevGames = []
        allprevGames = []
        #Save opponent Win Probability Based on FPI
      #  oppWinProb = fpi_values[oppTeamABR]
        #init team win 
        teamWinProb = 0
        #init Player Team
        playerTeam = ""

        if pposition == "File":

        # ---------- RUN MODEL ON LOAD OF FULL TEAMS --------- #

            kcqb = rosters["KAN"]["QB"]
            detqb = rosters["DET"]["QB"]
            detRbs = rosters["DET"]["RB"]
            kcRbs = rosters["KAN"]["RB"]
            detwrs = rosters["DET"]["WR"]
            kcwrs = rosters["KAN"]["WR"]
            kcTe = rosters["KAN"]["TE"]
            detTe = rosters["DET"]["TE"]


            print(kcqb)
            print(detRbs["1"])
            if len(detRbs) > 1:
                for rb in detRbs:
                    print("d")
         #           runRbProj("", "", "RB", "Kansas City Chiefs","KAN","Away", rb)
          #  else:
         #       runRbProj("", "", "RB", "Kansas City Chiefs","KAN","Away", detRbs["1"])
            
          #  print(detwrs.values)
            if len(detwrs) > 1:
                for wr in detwrs:
                    print(detwrs[wr])
                    runWRProj("", "", "WR", "Kansas City Chiefs","KAN","Away", detwrs[wr])
            else:
                runWRProj("", "", "WR", "Kansas City Chiefs","KAN","Away", detwrs["1"])


            print(all_projections)

        # ---------------------------------------- #
        # -----------------------------------------#
        # --------- RUNNING BACKS -----------------#
        # ---------------------------------------- #
        # -----------------------------------------#

        elif pposition == "RB":
            
            runRbProj(playerFirst, playerLast, pposition, oppTeam,oppTeamABR,loc,"")
            
            #window.contents_changed()    # Update the content of `sg.Column` after window.refresh()
        elif pposition == "WR" or pposition == "TE":
            
            # ---------------------------------------------#
            #----------------------------------------------#
            #--------------- WIDE RECIEVER ---------------- #
            #-----------------------------------------------#
            #-----------------------------------------------#

            runWRProj(playerFirst, playerLast, pposition, oppTeam,oppTeamABR,loc,"")

           
        elif pposition == "QB":

            # ------------------------------------------ #
            # ------------------------------------------ #
            # ------------- QUARTER BACKS -------------- # 
            # ------------------------------------------ #
            # ------------------------------------------ #
            runQBProj(playerFirst, playerLast, pposition, oppTeam,oppTeamABR,loc,"")

            
        else:
            print("Position Not Valid")
    if event == 'Cycle Layout':
        window[f'-COL{layout}-'].update(visible=False)
        layout = ((layout + 1) % 3) + 1
        window[f'-COL{layout}-'].update(visible=True)
    elif event in '123':
        window[f'-COL{layout}-'].update(visible=False)
        layout = int(event)
        window[f'-COL{layout}-'].update(visible=True)
    

window.close()

