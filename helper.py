from urlextract import URLExtract
extractor = URLExtract()
from wordcloud import WordCloud
import pandas as pd
from collections import Counter
import emoji
from textblob import TextBlob


def fetch_stats(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    #1. fetch no. of msgs
    num_messages = df.shape[0]


    #2. no. of words
    words = []
    for msg in df['message']:
        words.extend(msg.split(' '))


    # 3. no. of media
    num_of_media = df[df['message'] == '<Media omitted>\n'].shape[0]


    #4.no. of links
    links = []
    for msg in df['message']:
        links.extend(extractor.find_urls(msg))

    return num_messages, len(words), num_of_media, len(links)


def most_busy_users(df):
    x = df[df['user'] != 'group_notification']['user'].value_counts().head()
    # x = df['user'].value_counts().head()
    df = (df[df['user'] != 'group_notification']['user'].value_counts(normalize=True) * 100).round(2).reset_index().rename(
        columns={'user': 'Name', 'proportion': 'Percent'})
    return x,df


def create_word_cloud(selected_user,df):

    f = open('stop_hinglish.txt', 'r')
    stop_words = f.read()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']  # removed system msg
    temp = temp[temp['message'] != '<Media omitted>\n']  # removed media msg

    def remove_stop_words(message):
        y = []
        for word in message.lower().split():
            if word not in stop_words:
                y.append(word)

        return " ".join(y)

    wc = WordCloud(width=500,height=400,min_font_size=10,background_color='white')
    temp['message'] = temp['message'].apply(remove_stop_words)
    # df = df[df['message'] != '<Media omitted>\n']
    df_wc = wc.generate(temp['message'].str.cat(sep=' '))
    return df_wc


def most_comm_words(selected_user,df):
    f = open('stop_hinglish.txt', 'r')
    stop_words = f.read()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']   #removed system msg
    temp = temp[temp['message'] != '<Media omitted>\n']  # removed media msg

    words = []
    for msg in temp['message']:
        for word in msg.lower().split():
            if word not in stop_words:
                words.append(word)
    most_comm_df = pd.DataFrame(Counter(words).most_common(15))

    return most_comm_df


def emoji_helper(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    emojis = []
    for msg in df['message']:
        emojis.extend([c for c in msg if emoji.is_emoji(c)])

    emoji_df = pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))))

    return emoji_df

def monthly_timeline(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    timeline = df.groupby(['year', 'month', 'month_num']).count()['message'].reset_index()

    timeline = timeline.sort_values(['year', 'month_num'])

    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i] + "-" + str(timeline['year'][i]))
    timeline['time'] = time

    return timeline


def week_activity_map(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    return df['day_name'].value_counts()

def hour_activity_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    hour_count = df['hour'].value_counts().sort_index()

    # convert to 12-hour format labels
    labels = []
    for h in hour_count.index:
        if h == 0:
            labels.append('12 AM')
        elif h < 12:
            labels.append(f'{h} AM')
        elif h == 12:
            labels.append('12 PM')
        else:
            labels.append(f'{h-12} PM')

    hour_count.index = labels

    return hour_count


def sentiment_analysis(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    sentiments = []

    for msg in df['message']:
        polarity = TextBlob(msg).sentiment.polarity

        if polarity > 0:
            sentiments.append('Positive')
        elif polarity < 0:
            sentiments.append('Negative')
        else:
            sentiments.append('Neutral')

    df['sentiment'] = sentiments

    return df['sentiment'].value_counts()


def sentiment_timeline(selected_user, df):

    from textblob import TextBlob

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    df = df[df['user'] != 'group_notification']

    sentiments = []
    for msg in df['message']:
        polarity = TextBlob(msg).sentiment.polarity

        if polarity > 0:
            sentiments.append('Positive')
        elif polarity < 0:
            sentiments.append('Negative')
        else:
            sentiments.append('Neutral')

    df['sentiment'] = sentiments

    timeline = df.groupby(['only_date', 'sentiment']).size().unstack().fillna(0)

    return timeline

def sentiment_by_user(df):

    from textblob import TextBlob

    df = df[df['user'] != 'group_notification']

    users = []
    sentiments = []

    for i in range(df.shape[0]):
        msg = df.iloc[i]['message']
        user = df.iloc[i]['user']

        polarity = TextBlob(msg).sentiment.polarity

        if polarity > 0:
            sentiment = 'Positive'
        elif polarity < 0:
            sentiment = 'Negative'
        else:
            sentiment = 'Neutral'

        users.append(user)
        sentiments.append(sentiment)

    temp = pd.DataFrame({'user': users, 'sentiment': sentiments})

    result = temp.groupby(['user', 'sentiment']).size().unstack().fillna(0)

    return result




















































