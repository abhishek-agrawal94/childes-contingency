import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import math
import glob


def plot_manual_contingency():
    csv_files = glob.glob('./data/manually_annotated_data/*.csv')
    df_concat = pd.concat([pd.read_csv(f, delimiter=';') for f in csv_files], ignore_index=True)

    df_concat.drop(inplace=True, columns=['Unnamed: 0'])
    df_concat = df_concat[~df_concat['coherence_agreement'].isna()]
    df_concat['age'] = [x.split('-')[0] for x in df_concat['transcript_id']]
    X = ['non-contingent', 'ambiguous', 'contingent']

    mask = df_concat.speaker_code == 'MOT'
    col_name = 'speaker_code'
    df_concat.loc[mask, col_name] = 'PAR'

    mask = df_concat.speaker_code == 'FAT'
    col_name = 'speaker_code'
    df_concat.loc[mask, col_name] = 'PAR'

    df_parent = df_concat[df_concat.speaker_code == 'PAR']
    df_child = df_concat[df_concat.speaker_code == 'CHI']
    df_parent_32 = df_concat[(df_concat.speaker_code == 'PAR') & (df_concat.age == '32')]
    df_child_32 = df_concat[(df_concat.speaker_code == 'CHI') & (df_concat.age == '32')]
    df_parent_20 = df_concat[(df_concat.speaker_code == 'PAR') & (df_concat.age == '20')]
    df_child_20 = df_concat[(df_concat.speaker_code == 'CHI') & (df_concat.age == '20')]

    Y_child = df_child.coherence_agreement.value_counts(normalize=True, sort=False).sort_index()
    Y_parent = df_parent.coherence_agreement.value_counts(normalize=True, sort=False).sort_index()
    Y_child_32 = df_child_32.coherence_agreement.value_counts(normalize=True, sort=False).sort_index()
    Y_parent_32 = df_parent_32.coherence_agreement.value_counts(normalize=True, sort=False).sort_index()
    Y_child_20 = df_child_20.coherence_agreement.value_counts(normalize=True, sort=False).sort_index()
    Y_parent_20 = df_parent_20.coherence_agreement.value_counts(normalize=True, sort=False).sort_index()

    # print(Y_child)
    # print(Y_parent)

    X_axis = np.arange(len(X))

    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.bar(X_axis - 0.2, Y_child, 0.4, label='Child', color='cornflowerblue', edgecolor='black')
    ax1.bar(X_axis + 0.2, Y_parent, 0.4, label='Parent', color='orange', edgecolor='black')

    ax2.bar(X_axis - 0.3, Y_child_20, 0.2, label='Child-20', color='cornflowerblue', edgecolor='black')
    ax2.bar(X_axis + 0.1, Y_parent_20, 0.2, label='Parent-20', color='orange', edgecolor='black')
    ax2.bar(X_axis - 0.1, Y_child_32, 0.2, label='Child-32', color='cornflowerblue', hatch='/', edgecolor='black')
    ax2.bar(X_axis + 0.3, Y_parent_32, 0.2, label='Parent-32', color='orange', hatch='/', edgecolor='black')

    ax1.set(xlabel='Contingency', ylabel='Proportion')
    ax2.set(xlabel='Contingency', ylabel='Proportion')

    ax1.label_outer()
    ax2.label_outer()

    ax1.set_xticks(X_axis)
    ax1.set_xticklabels(X, rotation=60)
    ax2.set_xticks(X_axis)
    ax2.set_xticklabels(X, rotation=60)
    # plt.xlabel("Contingency")
    # plt.ylabel("Proportion")
    fig.suptitle('Proportion of contingency in adults & children')
    # plt.title("Proportion of contingency in adults & children")
    ax1.legend()
    ax2.legend()
    plt.tight_layout()
    plt.savefig("results/contingency_proportions_new_england_dual_plot.png", dpi=300)
    plt.show()


def plot_within_range_auto_annotation_results():
    with open("results/majority_vote_parent.csv") as fp:
        results_df = pd.read_csv(fp)

    data_cohere = results_df.dropna(subset=["coherence", "age"]).copy()

    def age_bin(age, num_months=2):
        return int((age + num_months / 2) / num_months) * num_months

    data_cohere["coherent"] = (data_cohere["coherence"] == 1).astype(int)
    data_cohere["ambiguous"] = (data_cohere["coherence"] == 0).astype(int)
    data_cohere["incoherent"] = (data_cohere["coherence"] == -1).astype(int)

    data_cohere["age"] = data_cohere.age.apply(age_bin).astype(int)

    data_cohere = data_cohere[data_cohere["age"].between(20, 32, inclusive="both")]

    with open("results/majority_vote_child.csv") as fp:
        results_df = pd.read_csv(fp)

    data_cohere_child = results_df.dropna(subset=["coherence", "age"]).copy()

    data_cohere_child["coherent"] = (data_cohere_child["coherence"] == 1).astype(int)
    data_cohere_child["ambiguous"] = (data_cohere_child["coherence"] == 0).astype(int)
    data_cohere_child["incoherent"] = (data_cohere_child["coherence"] == -1).astype(int)

    data_cohere_child["age"] = data_cohere_child.age.apply(age_bin).astype(int)

    data_cohere_child = data_cohere_child[data_cohere_child["age"].between(20, 32, inclusive="both")]

    sns.lineplot(data=data_cohere, x="age", y="coherent", errorbar="se", linestyle='solid', color='cornflowerblue')
    sns.lineplot(data=data_cohere, x="age", y="ambiguous", errorbar="se", linestyle='solid', color='orange')
    sns.lineplot(data=data_cohere, x="age", y="incoherent", errorbar="se", linestyle='solid', color='green')
    sns.lineplot(data=data_cohere_child, x="age", y="coherent", errorbar="se", linestyle='dashdot',
                 color='cornflowerblue')
    sns.lineplot(data=data_cohere_child, x="age", y="ambiguous", errorbar="se", linestyle='dashdot', color='orange')
    ax = sns.lineplot(data=data_cohere_child, x="age", y="incoherent", errorbar="se", linestyle='dashdot',
                      color='green')
    plt.ylabel("proportion")
    plt.xlabel("age (months)")
    plt.xticks([20, 22, 24, 26, 28, 30, 32])
    plt.legend(handles=ax.lines, loc='upper left',
               labels=["contingent parent", "ambiguous parent", "non-contingent parent", "contingent child",
                       "ambiguous child", "non-contingent child"], ncol=2)
    plt.ylim(-0.1, 1.1)
    plt.tight_layout()
    plt.savefig("results/annotations_childes_20_32_ages.png", dpi=300)


def plot_outside_range_auto_annotation_results():
    with open("results/majority_vote_parent.csv") as fp:
        results_df = pd.read_csv(fp)

    data_cohere = results_df.dropna(subset=["coherence", "age"]).copy()

    def age_bin(age, num_months=3):
        return int((age + num_months / 2) / num_months) * num_months

    data_cohere["coherent"] = (data_cohere["coherence"] == 1).astype(int)
    data_cohere["ambiguous"] = (data_cohere["coherence"] == 0).astype(int)
    data_cohere["incoherent"] = (data_cohere["coherence"] == -1).astype(int)

    data_cohere["age"] = data_cohere.age.apply(age_bin).astype(int)

    data_cohere = data_cohere[data_cohere["age"].between(20, 64, inclusive="both")]

    with open("results/majority_vote_child.csv") as fp:
        results_df = pd.read_csv(fp)

    data_cohere_child = results_df.dropna(subset=["coherence", "age"]).copy()

    data_cohere_child["coherent"] = (data_cohere_child["coherence"] == 1).astype(int)
    data_cohere_child["ambiguous"] = (data_cohere_child["coherence"] == 0).astype(int)
    data_cohere_child["incoherent"] = (data_cohere_child["coherence"] == -1).astype(int)

    data_cohere_child["age"] = data_cohere_child.age.apply(age_bin).astype(int)

    data_cohere_child = data_cohere_child[data_cohere_child["age"].between(20, 64, inclusive="both")]

    # plt.figure(1, figsize=(8, 10))

    sns.lineplot(data=data_cohere, x="age", y="coherent", errorbar="se", linestyle='solid', color='cornflowerblue')
    sns.lineplot(data=data_cohere, x="age", y="ambiguous", errorbar="se", linestyle='solid', color='orange')
    sns.lineplot(data=data_cohere, x="age", y="incoherent", errorbar="se", linestyle='solid', color='green')
    sns.lineplot(data=data_cohere_child, x="age", y="coherent", errorbar="se", linestyle='dashdot',
                 color='cornflowerblue')
    sns.lineplot(data=data_cohere_child, x="age", y="ambiguous", errorbar="se", linestyle='dashdot', color='orange')
    ax = sns.lineplot(data=data_cohere_child, x="age", y="incoherent", errorbar="se", linestyle='dashdot',
                      color='green')
    plt.ylabel("proportion")
    plt.xlabel("age (months)")
    plt.xticks([24, 30, 36, 42, 48, 54, 60])
    plt.ylim(-0.1, 1.1)
    plt.legend(handles=ax.lines, loc='upper left',
               labels=["contingent parent", "ambiguous parent", "non-contingent parent",
                       "contingent child", "ambiguous child", "non-contingent child"], ncol=2)
    plt.tight_layout()
    plt.savefig("results/annotations_childes_20_64_ages.png", dpi=300)


