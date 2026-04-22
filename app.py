import streamlit as st
import preprocessor, helper
import matplotlib.pyplot as plt
import plotly.express as px

plt.style.use('ggplot')
plt.rcParams['font.family'] = 'Segoe UI Emoji'

st.set_page_config(
    page_title='ChatInsight',
    layout='wide',
    page_icon='📈',
    initial_sidebar_state='expanded'
)

st.markdown('# _Welcome to Whatsapp Chat Analysis:_')
st.sidebar.title('ChatInsight')

uploaded_file = st.sidebar.file_uploader("Choose a file")

# ================= MAIN =================
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    df = preprocessor.preprocess(data)

    # SAFE user list
    userlist = df['user'].unique().tolist()
    if 'group_notification' in userlist:
        userlist.remove('group_notification')

    userlist.sort()
    userlist.insert(0, 'Overall')

    selected_user = st.sidebar.selectbox('Show analysis wrt :', userlist)

    # ================= ANALYSIS =================
    if st.sidebar.button('Show Analysis'):
        num_messages, words, num_of_media, num_of_links = helper.fetch_stats(selected_user, df)

        st.title('Top Statistics')
        col1, col2 = st.columns(2)
        col3, col4 = st.columns(2)

        with col1:
            st.header('Total Messages')
            st.title(num_messages)

        with col2:
            st.header('Total Words')
            st.title(words)

        with col3:
            st.header('Media Shared')
            st.title(num_of_media)

        with col4:
            st.header('Links Shared')
            st.title(num_of_links)

        # Monthly timeline
        st.title('Monthly Timeline')
        timeline = helper.monthly_timeline(selected_user, df)

        fig = px.line(
            timeline,
            x='time',
            y='message',
            markers=True,
            title='Messages Over Time',
            color_discrete_sequence=['#636EFA']
        )

        fig.update_layout(
            title_x=0.5,
            xaxis_title='Month',
            yaxis_title='Number of Messages',
            hovermode='x unified',
            xaxis=dict(rangeslider=dict(visible=True))
        )

        fig.update_xaxes(tickangle=90)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': 'hover'})

        # Activity Chart
        st.title('Activity Chart')
        col1, col2 = st.columns(2)

        with col1:
            busy_day = helper.week_activity_map(selected_user, df)

            fig = px.bar(
                busy_day,
                x=busy_day.index,
                y=busy_day.values,
                title='Most Busy Day'
            )

            fig.update_layout(
                title_x=0.5,
                xaxis_title='Day of the Week',
                yaxis_title='Number of Activities',
                autosize=True,
                margin=dict(l=10, r=10, t=40, b=10)
            )

            fig.update_xaxes(tickangle=90)
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': 'hover'})

        with col2:
            busy_hour = helper.hour_activity_map(selected_user, df)

            fig = px.bar(
                busy_hour,
                x=busy_hour.index,
                y=busy_hour.values,
                title='Most Busy Hour'
            )

            fig.update_layout(
                title_x=0.5,
                xaxis_title='Hour of Day',
                yaxis_title='Number of Messages',
                autosize=True,
                margin=dict(l=10, r=10, t=40, b=10)
            )

            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': 'hover'})

        # Most Busy Users
        if selected_user == 'Overall':
            st.title('Most Busy Users')
            col1, col2 = st.columns(2)

            with col1:
                x, new_df = helper.most_busy_users(df)

                bar_df = x.reset_index()
                bar_df.columns = ['user', 'count']
                bar_df['user'] = bar_df['user'].astype(str)

                fig = px.bar(bar_df, x='user', y='count', color_discrete_sequence=['#bdc3c7'])
                fig.update_layout(
                    xaxis_title='Users',
                    yaxis_title='Number of Messages'
                )
                fig.update_xaxes(type='category', tickangle=90)

                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': 'hover'})

            with col2:
                st.dataframe(new_df, use_container_width=True)

        # Wordcloud
        st.title('Frequency of Words:')
        df_wc = helper.create_word_cloud(selected_user, df)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.imshow(df_wc)
        ax.axis('off')  # FIX (UI)
        st.pyplot(fig)

        # Most Common words
        st.title('Most Common Words')
        most_comm_df = helper.most_comm_words(selected_user, df)

        fig = px.bar(
            most_comm_df,
            x=most_comm_df[1],
            y=most_comm_df[0],
            orientation='h',
            title='Frequent Words (15)',
            color_discrete_sequence=['#636EFA']
        )

        fig.update_layout(
            title_x=0.5,
            xaxis_title='Words',
            yaxis_title='Number of Activities',
            yaxis=dict(autorange="reversed"),
            autosize=True,
            margin=dict(l=10, r=10, t=40, b=10)
        )

        # fig.update_xaxes(tickangle=90)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': 'hover'})

        # Emoji
        emoji_df = helper.emoji_helper(selected_user, df)
        st.title('Emoji Analysis:')

        st.subheader("Emoji Table")
        st.dataframe(emoji_df, use_container_width=True)

        if not emoji_df.empty:
            emoji_df.columns = ['emoji', 'count']
            top_emoji_df = emoji_df.head(10)

            fig = px.bar(
                top_emoji_df,
                x='count',
                y='emoji',
                orientation='h',
                title='Top Emojis'
            )

            fig.update_layout(
                title_x=0.5,
                xaxis_title='Count',
                yaxis_title='Emoji',
                margin=dict(l=20, r=20, t=40, b=20)
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("No emojis found")

        # Sentiment
        st.title('Sentiment Analysis')
        sentiment_df = helper.sentiment_analysis(selected_user, df)

        fig = px.bar(
            x=sentiment_df.index,
            y=sentiment_df.values,
            color=sentiment_df.index,
            title='Sentiment Distribution',
            color_discrete_map={
                'Positive': 'green',
                'Negative': 'red',
                'Neutral': 'gray'
            }
        )

        fig.update_layout(
            title_x=0.5,
            xaxis_title='Sentiment',
            yaxis_title='Count',
            autosize=True,
            margin=dict(l=10, r=10, t=40, b=10)
        )

        st.plotly_chart(fig, use_container_width=True)

        # Sentiment Timeline
        st.title('Sentiment Over Time')
        sent_timeline = helper.sentiment_timeline(selected_user, df)

        fig = px.line(
            sent_timeline,
            x=sent_timeline.index,
            y=sent_timeline.columns,
            title='Sentiment Trend Over Time',
            color_discrete_map={
                'Positive': 'green',
                'Negative': 'red',
                'Neutral': 'gray'
            }
        )

        fig.update_layout(
            title_x=0.5,
            xaxis_title='Months',
            yaxis_title='Number of Activities',
            xaxis=dict(rangeslider=dict(visible=True)),
            autosize=True,
            margin=dict(l=10, r=10, t=40, b=10)
        )

        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': 'hover'})

        # Sentiment by User
        if selected_user == 'Overall':
            st.title('User Sentiment Analysis')
            user_sent_df = helper.sentiment_by_user(df)

            user_sent_df.index = user_sent_df.index.astype(str)

            fig = px.bar(
                user_sent_df,
                x=user_sent_df.index,
                y=user_sent_df.columns,
                barmode='group',
                title='User-wise Sentiment',
                color_discrete_map={
                    'Positive': 'green',
                    'Negative': 'red',
                    'Neutral': 'gray'
                }
            )

            fig.update_layout(
                title_x=0.5,
                autosize=True,
                margin=dict(l=10, r=10, t=40, b=10)
            )

            fig.update_xaxes(type='category', tickangle=90)
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': 'hover'})

        # Insights
        if selected_user == 'Overall':
            st.title('Key Insights')
            user_sent_df = helper.sentiment_by_user(df)

            most_positive = user_sent_df['Positive'].idxmax() if 'Positive' in user_sent_df else "N/A"
            most_negative = user_sent_df['Negative'].idxmax() if 'Negative' in user_sent_df else "N/A"
            most_active = df[df['user'] != 'group_notification']['user'].value_counts().idxmax()
            peak_day = df['day_name'].value_counts().idxmax()

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(label="Most Positive 😊", value=most_positive)

            with col2:
                st.metric(label="Most Negative 😡", value=most_negative)

            with col3:
                st.metric(label="Most Active 💬", value=most_active)

            with col4:
                st.metric(label="Peak Day 📅", value=peak_day)

    # ================= SMART ANALYZER =================
    st.title("Smart Query Analyzer")

    query_word = st.text_input("Enter word (example: sorry)")

    if st.button("Count Word"):
        if query_word:
            result = helper.count_word_usage(selected_user, df, query_word)

            if selected_user == 'Overall':
                st.success(f"💬 '{query_word}' was used {result} times in the chat")
            else:
                st.success(f"💬 {selected_user} used '{query_word}' {result} times")

    if st.button("Who sends longer messages?"):
        result = helper.avg_message_length(df)

        top_user = result.idxmax()
        bottom_user = result.idxmin()

        st.success(f"{top_user} tends to send longer messages on average")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Longest Messages", top_user)

        with col2:
            st.metric("Shortest Messages", bottom_user)

    if st.button("Who replies faster?"):
        result = helper.reply_speed(df)

        fastest_user = result.idxmin()
        slowest_user = result.idxmax()

        st.success(f"⚡ {fastest_user} tends to reply faster in conversations")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Fastest Replier", fastest_user)

        with col2:
            st.metric("Slowest Replier", slowest_user)

        st.caption("Average reply time in hours")
        st.dataframe(result)

    if st.button("Who starts conversation more?"):
        result = helper.conversation_starter(df)
        top_user = result.idxmax()

        st.success(f"{top_user} usually initiates conversations more often")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Top Conversation Starter", top_user)

        with col2:
            st.metric("Starts Count", int(result.max()))

    if st.button("Who ends conversation more?"):
        result = helper.conversation_ender(df)
        top_user = result.idxmax()

        st.success(f"{top_user} often ends conversations")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Top Conversation Ender", top_user)

        with col2:
            st.metric("Ends Count", int(result.max()))

    if st.button("On which day we talked the longest?"):
        result = helper.longest_chat_day(df)

        top_day = result.idxmax()
        top_count = result.max()

        st.success(f"📅 Your most active chat day was {top_day} with {top_count} messages")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Top Chat Day", str(top_day))

        with col2:
            st.metric("Messages", int(top_count))
