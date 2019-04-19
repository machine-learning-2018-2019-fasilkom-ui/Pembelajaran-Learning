from sklearn.preprocessing import RobustScaler

import csv
import pandas as pd

clean_table = pd.DataFrame()

appendix = pd.read_csv("dataset/winrate6.csv", names=["winrate_home", "winrate_away"])

clean_table.loc[:, "winrate_home"] = pd.Series(appendix.loc[:, "winrate_home"])
clean_table.loc[:, "winrate_away"] = pd.Series(appendix.loc[:, "winrate_away"])

appendix = pd.read_csv("dataset/ranks6.csv", names=["rank_home", "rank_away"])

clean_table.loc[:, "rank_home"] = pd.Series(appendix.loc[:, "rank_home"])
clean_table.loc[:, "rank_away"] = pd.Series(appendix.loc[:, "rank_away"])

appendix = pd.read_csv("dataset/wins6.csv", names=["head_to_head"])

clean_table.loc[:, "head_to_head"] = pd.Series(appendix.loc[:, "head_to_head"])

with open('dataset/odds6.csv', 'rt') as f:
    reader = csv.reader(f)
    probs_list = list(reader)

probs_dict = {0: [], 1: [], 2: []}

for probs in probs_list:
    for i in range(len(probs)):
        if float(probs[i]) != 0:
            probs_dict[i].append(1 / float(probs[i]))
        else:
            probs_dict[i].append(1)

clean_table.loc[:, "home_prob"] = pd.Series(probs_dict[0])
clean_table.loc[:, "draw_prob"] = pd.Series(probs_dict[1])
clean_table.loc[:, "away_prob"] = pd.Series(probs_dict[2])

transformer = RobustScaler().fit(clean_table.values)
clean_table = pd.DataFrame(transformer.transform(clean_table.values),
                           columns=list(clean_table))

with open('dataset/fresults6.csv', 'rt') as f:
    reader = csv.reader(f)
    final_score = list(reader)

game_conclusion = []
for i in range(len(final_score)):
    if final_score[i][0] > final_score[i][1]:
        game_conclusion.append("HOME")
    elif final_score[i][0] == final_score[i][1]:
        game_conclusion.append("DRAW")
    else:
        game_conclusion.append("AWAY")

clean_table.loc[:, 'game_conclusion'] = pd.Series(game_conclusion)

clean_table.to_csv("clean_table.csv", index=False, quoting=csv.QUOTE_NONE)
