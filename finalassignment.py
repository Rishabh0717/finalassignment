import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV file
df = pd.read_csv('data/deliveries-2.csv')

# 1. Basic Statistics
print("=== Basic Statistics ===")
print("\nDataset Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nSummary Statistics:")
print(df.describe())

# Calculate additional statistics
total_matches = df['match_id'].nunique()
total_innings = df['inning'].nunique()
total_runs = df['total_runs'].sum()
total_wickets = df['player_dismissed'].notna().sum()
total_extras = df['extra_runs'].sum()

print(f"\nTotal Matches: {total_matches}")
print(f"Total Innings: {total_innings}")
print(f"Total Runs: {total_runs}")
print(f"Total Wickets: {total_wickets}")
print(f"Total Extras: {total_extras}")

plt.figure(figsize=(12, 8))
run_types = {
    'Batsman Runs': df['batsman_runs'].sum(),
    'Wide Runs': df['wide_runs'].sum(),
    'Bye Runs': df['bye_runs'].sum(),
    'Legbye Runs': df['legbye_runs'].sum(),
    'No Ball Runs': df['noball_runs'].sum(),
    'Penalty Runs': df['penalty_runs'].sum()
}
plt.pie(run_types.values(), labels=run_types.keys(), autopct='%1.1f%%',
        colors=sns.color_palette('husl'))
plt.title('Distribution of Run Types')
plt.axis('equal')
plt.show()

print("\nInsight 1: Run Distribution")
print(f"Total runs scored: {sum(run_types.values())}")
print(f"Percentage of runs from batsmen: {(run_types['Batsman Runs']/sum(run_types.values())*100):.2f}%")
print(f"Percentage of extras: {((sum(run_types.values()) - run_types['Batsman Runs'])/sum(run_types.values())*100):.2f}%")


plt.figure(figsize=(15, 6))
top_batsmen = df.groupby('batsman')['batsman_runs'].sum().sort_values(ascending=False).head(10)
sns.barplot(x=top_batsmen.index, y=top_batsmen.values)
plt.title('Top 10 Batsmen by Runs Scored')
plt.xticks(rotation=45)
plt.xlabel('Batsman')
plt.ylabel('Total Runs')
plt.show()

print("\nInsight 2: Top Batsmen")
print("Top 5 run scorers:")
for idx, (batsman, runs) in enumerate(top_batsmen.head().items(), 1):
    print(f"{idx}. {batsman}: {runs} runs")

plt.figure(figsize=(12, 6))
dismissal_counts = df['dismissal_kind'].value_counts()
sns.barplot(x=dismissal_counts.index, y=dismissal_counts.values)
plt.title('Types of Dismissals')
plt.xticks(rotation=45)
plt.xlabel('Dismissal Type')
plt.ylabel('Count')
plt.show()

print("\nInsight 3: Dismissal Patterns")
print("Most common types of dismissals:")
for idx, (dismissal, count) in enumerate(dismissal_counts.head().items(), 1):
    print(f"{idx}. {dismissal}: {count} times")

plt.figure(figsize=(12, 6))
team_runs = df.groupby('batting_team')['total_runs'].sum().sort_values(ascending=False)
sns.barplot(x=team_runs.index, y=team_runs.values)
plt.title('Total Runs by Team')
plt.xticks(rotation=45)
plt.xlabel('Team')
plt.ylabel('Total Runs')
plt.show()

print("\nInsight 4: Team Performance")
print("Team-wise total runs:")
for idx, (team, runs) in enumerate(team_runs.items(), 1):
    print(f"{idx}. {team}: {runs} runs")


plt.figure(figsize=(15, 6))
over_runs = df.groupby('over')['total_runs'].mean()
plt.plot(over_runs.index, over_runs.values, marker='o')
plt.title('Average Runs per Over')
plt.xlabel('Over Number')
plt.ylabel('Average Runs')
plt.grid(True)
plt.show()

print("\nInsight 5: Over-wise Analysis")
print(f"Most productive over: Over {over_runs.idxmax()} (Avg. {over_runs.max():.2f} runs)")
print(f"Least productive over: Over {over_runs.idxmin()} (Avg. {over_runs.min():.2f} runs)")


plt.figure(figsize=(15, 8))
runs_heatmap = df.pivot_table(index='inning', columns='over',
                             values='total_runs', aggfunc='mean')
sns.heatmap(runs_heatmap, cmap='YlOrRd', annot=True, fmt='.1f')
plt.title('Average Runs per Over by Innings')
plt.xlabel('Over Number')
plt.ylabel('Innings')
plt.show()

plt.figure(figsize=(10, 6))
extras_by_team = df.groupby('bowling_team')[['wide_runs', 'noball_runs', 'bye_runs', 'legbye_runs']].sum()
extras_by_team.plot(kind='bar', stacked=True)
plt.title('Extra Runs Conceded by Teams')
plt.xlabel('Team')
plt.ylabel('Runs')
plt.legend(title='Extra Types', bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.show()

print("\nInsight 6: Extras Analysis")
total_extras = extras_by_team.sum().sum()
print(f"Total extras in the tournament: {total_extras}")
print("\nTeam-wise extras conceded:")
for team in extras_by_team.index:
    print(f"{team}: {extras_by_team.loc[team].sum()} runs")


plt.figure(figsize=(15, 6))
sns.boxplot(x='batting_team', y='total_runs', data=df)
plt.title('Run Distribution by Teams')
plt.xticks(rotation=45)
plt.xlabel('Team')
plt.ylabel('Runs per Ball')
plt.show()

print("\nInsight 7: Overall Analysis")
print(f"Total matches analyzed: {df['match_id'].nunique()}")
print(f"Total overs bowled: {len(df['over'].unique())}")
print(f"Average runs per over: {df.groupby('over')['total_runs'].mean().mean():.2f}")
print(f"Highest team score: {df.groupby(['match_id', 'batting_team'])['total_runs'].sum().max()}")

summary_stats = pd.DataFrame({
    'Total Matches': [df['match_id'].nunique()],
    'Total Runs': [df['total_runs'].sum()],
    'Total Wickets': [df['player_dismissed'].notna().sum()],
    'Average Runs/Over': [df.groupby('over')['total_runs'].mean().mean()],
    'Total Extras': [total_extras]
})

print("\nSummary Statistics:")
print(summary_stats.T)