from urlextract import URLExtract
extractor = URLExtract()

from wordcloud import WordCloud
import pandas as pd
from collections import Counter
import emoji
from textblob import TextBlob
import re


def fetch_stats(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    num_messages = df.shape[0]

    words = []
    for msg in df['message']:
        words.extend(msg.split())

    num_of_media = df[df['message'] == '<Media omitted>\n'].shape[0]

    links = []
    for msg in df['message']:
        links.extend(extractor.find_urls(msg))

    return num_messages, len(words), num_of_media, len(links)


def most_busy_users(df):
    x = df[df['user'] != 'group_notification']['user'].value_counts().head()

    df_percent = (
        df[df['user'] != 'group_notification']['user']
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
        .reset_index()
        .rename(columns={'user': 'Name', 'proportion': 'Percent'})
    )

    return x, df_percent


def create_word_cloud(selected_user, df):
    with open('stop_hinglish.txt', 'r') as f:
        stop_words = f.read().split()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[
        (df['user'] != 'group_notification') &
        (df['message'] != '<Media omitted>\n') &
        (~df['message'].str.contains('deleted', case=False, na=False))
    ].copy()

    def clean_text(message):
        message = re.sub(r'[^a-zA-Z\s]', '', message)

        return " ".join(
            word for word in message.lower().split()
            if word not in stop_words and len(word) > 2
        )

    temp['message'] = temp['message'].apply(clean_text)

    wc = WordCloud(width=500, height=400, min_font_size=10, background_color='white')
    return wc.generate(temp['message'].str.cat(sep=' '))


def most_comm_words(selected_user, df):
    with open('stop_hinglish.txt', 'r') as f:
        stop_words = f.read().split()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[
        (df['user'] != 'group_notification') &
        (df['message'] != '<Media omitted>\n') &
        (~df['message'].str.contains('deleted', case=False, na=False))
    ]

    words = []
    for msg in temp['message']:
        msg = re.sub(r'[^a-zA-Z\s]', '', msg)

        for word in msg.lower().split():
            if word not in stop_words and len(word) > 2:
                words.append(word)

    return pd.DataFrame(Counter(words).most_common(15))


def emoji_helper(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    emojis = [
        char
        for msg in df['message']
        for char in msg
        if emoji.is_emoji(char)
    ]

    if not emojis:
        return pd.DataFrame(columns=['emoji', 'count'])

    return pd.DataFrame(Counter(emojis).most_common(), columns=['emoji', 'count'])


def monthly_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    timeline = df.groupby(['year', 'month', 'month_num']).count()['message'].reset_index()
    timeline = timeline.sort_values(['year', 'month_num'])

    timeline['time'] = timeline['month'] + "-" + timeline['year'].astype(str)

    return timeline


def week_activity_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['day_name'].value_counts()


def hour_activity_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    hour_count = df['hour'].value_counts().sort_index()

    labels = []
    for h in hour_count.index:
        if h == 0:
            labels.append('12 AM')
        elif h < 12:
            labels.append(f'{h} AM')
        elif h == 12:
            labels.append('12 PM')
        else:
            labels.append(f'{h - 12} PM')

    hour_count.index = labels
    return hour_count


def sentiment_analysis(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    df = df.copy()

    df['sentiment'] = df['message'].apply(
        lambda msg: 'Positive' if TextBlob(msg).sentiment.polarity > 0
        else 'Negative' if TextBlob(msg).sentiment.polarity < 0
        else 'Neutral'
    )

    return df['sentiment'].value_counts()


def sentiment_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    df = df[df['user'] != 'group_notification'].copy()

    df['sentiment'] = df['message'].apply(
        lambda msg: 'Positive' if TextBlob(msg).sentiment.polarity > 0
        else 'Negative' if TextBlob(msg).sentiment.polarity < 0
        else 'Neutral'
    )

    return df.groupby(['only_date', 'sentiment']).size().unstack().fillna(0)


def sentiment_by_user(df):
    df = df[df['user'] != 'group_notification'].copy()

    df['sentiment'] = df['message'].apply(
        lambda msg: 'Positive' if TextBlob(msg).sentiment.polarity > 0
        else 'Negative' if TextBlob(msg).sentiment.polarity < 0
        else 'Neutral'
    )

    return df.groupby(['user', 'sentiment']).size().unstack().fillna(0)


def count_word_usage(selected_user, df, word):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    df = df[
        (df['user'] != 'group_notification') &
        (df['message'] != '<Media omitted>\n')
    ]

    return int(df['message'].str.lower().str.count(word.lower()).sum())


def avg_message_length(df):
    temp = df[
        (df['user'] != 'group_notification') &
        (df['message'] != '<Media omitted>\n')
    ].copy()

    temp['msg_length'] = temp['message'].str.split().str.len()

    return temp.groupby('user')['msg_length'].mean().sort_values(ascending=False)


def reply_speed(df):
    temp = df[
        (df['user'] != 'group_notification') &
        (df['message'] != '<Media omitted>\n')
    ].copy()

    temp = temp.sort_values('date')

    temp['prev_user'] = temp['user'].shift(1)
    temp['prev_date'] = temp['date'].shift(1)

    temp = temp[temp['user'] != temp['prev_user']]

    temp['reply_time_hours'] = (temp['date'] - temp['prev_date']).dt.total_seconds() / 3600
    temp = temp[temp['reply_time_hours'] < 2]

    return temp.groupby('user')['reply_time_hours'].mean().sort_values().round(2)


def conversation_starter(df):
    temp = df[
        (df['user'] != 'group_notification') &
        (df['message'] != '<Media omitted>\n')
    ].copy()

    temp = temp.sort_values('date')

    temp['prev_date'] = temp['date'].shift(1)
    temp['gap_hours'] = (temp['date'] - temp['prev_date']).dt.total_seconds() / 3600

    return temp[temp['gap_hours'] > 2]['user'].value_counts()


def conversation_ender(df):
    temp = df[
        (df['user'] != 'group_notification') &
        (df['message'] != '<Media omitted>\n')
    ].copy()

    temp = temp.sort_values('date')

    temp['next_date'] = temp['date'].shift(-1)
    temp['gap_hours'] = (temp['next_date'] - temp['date']).dt.total_seconds() / 3600

    return temp[temp['gap_hours'] > 2]['user'].value_counts()


def longest_chat_day(df):
    temp = df[
        (df['user'] != 'group_notification') &
        (df['message'] != '<Media omitted>\n')
    ]

    return temp.groupby('only_date').size().sort_values(ascending=False)
