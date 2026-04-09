import streamlit as st
import preprocessor,helper
import matplotlib.pyplot as plt
plt.style.use('ggplot')

plt.rcParams['font.family'] = 'Segoe UI Emoji'
import plotly.express as px


st.set_page_config(page_title='Whatsapp Chat Analysis',layout='wide',page_icon='📈',initial_sidebar_state='expanded')
st.markdown('# _Welcome to Whatsapp Chat Analysis:_')


st.sidebar.title('Whatsapp Chat Analyzer')

uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    df = preprocessor.preprocess(data)

    # Fetch unique users
    userlist = df['user'].unique().tolist()

    userlist.remove('group_notification')
    userlist.sort()
    userlist.insert(0,'Overall')

    selected_user = st.sidebar.selectbox('Show analysis wrt :', userlist)

    if st.sidebar.button('Show Analysis'):
        num_messages,words,num_of_media,num_of_links = helper.fetch_stats(selected_user,df)
        st.title('Top Statistics')
        # col1, col2, col3, col4 = st.columns(4)
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
        timeline = helper.monthly_timeline(selected_user,df)

        fig = px.line(
            timeline,
            x='time',
            y='message',
            markers=True,
            title='Messages Over Time',
            color_discrete_sequence= ['#636EFA']
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


        # Activity Map
        st.title('Activity Map')
        col1,col2 = st.columns(2)

        with col1:
            busy_day = helper.week_activity_map(selected_user,df)

            fig = px.bar(
                busy_day,
                x=busy_day.index,
                y=busy_day.values,
                title = 'Most Busy Day'
            )
            fig.update_layout(
                title_x = 0.5,
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


        # 5. finding the busiest user in the grp (group level)
        if selected_user == 'Overall':
            st.title('Most Busy Users')
            x,new_df = helper.most_busy_users(df)

            col1,col2 = st.columns(2)

            with col1:
                fig = px.bar(
                    x,
                    x = x.index,
                    y = x.values,
                    color_discrete_sequence=['#bdc3c7']
                )
                fig.update_layout(
                    xaxis_title='Users',
                    yaxis_title='Number of Activities',
                    autosize=True,
                    margin=dict(l=10, r=10, t=40, b=10)
                )

                fig.update_xaxes(tickangle=90)
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': 'hover'})

            with col2:
                st.dataframe(new_df)

        # Wordcloud
        st.title('Frequency of Words:')
        df_wc = helper.create_word_cloud(selected_user,df)
        fig,ax = plt.subplots(figsize=(6,4))
        ax.imshow(df_wc)
        st.pyplot(fig)

        # Most Common words
        st.title('Most Common Words')
        most_comm_df = helper.most_comm_words(selected_user,df)
        fig = px.bar(
            most_comm_df,
            x = most_comm_df[1],
            y = most_comm_df[0],
            orientation= 'h',
            title = 'Frequent Words (15)',
            color_discrete_sequence = ['#636EFA']
        )

        fig.update_layout(
            title_x = 0.5,
            xaxis_title='Words',
            yaxis_title='Number of Activities',
            yaxis=dict(autorange="reversed"),
            autosize=True,
            margin=dict(l=10, r=10, t=40, b=10)
        )

        fig.update_xaxes(tickangle=90)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': 'hover'})


        # emoji analysis
        emoji_df = helper.emoji_helper(selected_user,df)
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


        # Sentiment Analysis
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
            fig.update_xaxes(tickangle=90)

            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': 'hover'})


        # Insights
        if selected_user == 'Overall':
            st.title('Key Insights')
            user_sent_df = helper.sentiment_by_user(df)

            most_positive = user_sent_df['Positive'].idxmax() if 'Positive' in user_sent_df else "N/A"
            most_negative = user_sent_df['Negative'].idxmax() if 'Negative' in user_sent_df else "N/A"
            most_active = df['user'].value_counts().idxmax()
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





























































































