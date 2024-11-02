import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import scipy.stats as stats
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


df = pd.read_csv("ecommerce_customer_behavior_dataset.csv")

def mean_median_mode():
    mean_age = df['Age'].mean()
    median_age = df['Age'].median()
    mode_age = df['Age'].mode()[0]
    measures = ['Mean', 'Median', 'Mode']
    values = [mean_age, median_age, mode_age]

    plt.figure(figsize=(8, 6))
    plt.bar(measures, values, color=['skyblue', 'lightcoral', 'lightgreen'])
    plt.xlabel("Measure")
    plt.ylabel("Age")
    plt.title("Mean, Median, and Mode of Age")

    for i, v in enumerate(values):
        plt.text(i, v + 0.5, str(round(v, 2)), ha='center')

    st.pyplot(plt)
def variance():
    variance_purchase = df['Purchase Amount ($)'].var()
    std_dev_purchase = df['Purchase Amount ($)'].std()

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    sns.histplot(df['Purchase Amount ($)'], ax=axes[0], kde=True)
    axes[0].axvline(df['Purchase Amount ($)'].mean(), color='red', linestyle='dashed', linewidth=2, label='Mean')
    axes[0].axvline(df['Purchase Amount ($)'].mean() + std_dev_purchase, color='green', linestyle='dashed', linewidth=2,
                    label='+1 Std Dev')
    axes[0].axvline(df['Purchase Amount ($)'].mean() - std_dev_purchase, color='green', linestyle='dashed', linewidth=2,
                    label='-1 Std Dev')
    axes[0].set_title('Distribution of Purchase Amount with Standard Deviation')
    axes[0].legend()

    z_scores_purchase = stats.zscore(df['Purchase Amount ($)'])

    sns.histplot(z_scores_purchase, ax=axes[1], kde=True)
    axes[1].set_title('Z-scores of Purchase Amount')

    plt.tight_layout()
    st.pyplot(fig)

    st.write(f"Variance of Purchase Amount: {variance_purchase}")
    st.write(f"Standard Deviation of Purchase Amount: {std_dev_purchase}")

def top_products(df):
    top_products = df['Product Category'].value_counts().reset_index()
    top_products.columns = ['Product Category', 'Count']

    top_3_products = top_products.head(3)


    fig = px.bar(top_3_products,
                 x='Product Category',
                 y='Count',
                 title='Top 3 Product Categories Based on Number of Purchases',
                 labels={'Count': 'Number of Purchases'},
                 color='Product Category')


    st.title("Top 3 Product Categories")
    st.write("This chart shows the top 3 product categories based on the number of purchases.")


    st.plotly_chart(fig)
def return_customer():
    product_return_rate = df.groupby('Product Category')['Return Customer'].mean()
    return_customers = df[df['Return Customer'] == True]['Customer ID'].nunique()
    non_return_customers = df[df['Return Customer'] == False]['Customer ID'].nunique()

    st.write(f'Number of Return Customers: {return_customers}')
    st.write(f'Number of Non-Return Customers: {non_return_customers}')

    customer_types = ['Return Customers', 'Non-Return Customers']
    customer_counts = [return_customers, non_return_customers]

    plt.figure(figsize=(8, 6))
    plt.bar(customer_types, customer_counts, color=['skyblue', 'lightcoral'])
    plt.xlabel('Customer Type')
    plt.ylabel('Number of Customers')
    plt.title('Return vs. Non-Return Customers')

    for i, v in enumerate(customer_counts):
        plt.text(i, v + 0.5, str(v), ha='center')

    st.pyplot(plt)
    top_5_products = product_return_rate.sort_values(ascending=False).head(5).reset_index()
    top_5_products.columns = ['Product Category', 'Return Rate']

    fig_top_products = px.bar(
        top_5_products,
        x='Product Category',
        y='Return Rate',
        title='Top 5 Product Categories with the Highest Return Rate',
        labels={'Return Rate': 'Return Rate'},
        color='Return Rate',
        color_continuous_scale=px.colors.sequential.Viridis
    )

    fig_top_products.update_traces(texttemplate='%{y:.2%}', textposition='outside')
    fig_top_products.update_layout(yaxis_tickformat='%')

    st.plotly_chart(fig_top_products)

def review_score():
    avg_review_score = df['Review Score (1-5)'].mean()
    print(f'Average review score is : {avg_review_score}')

    avg_review_per_product = df.groupby('Product Category')['Review Score (1-5)'].mean().reset_index()
    avg_review_per_product.columns = ['Product Category', 'Average Review Score']

    best_product = avg_review_per_product.loc[avg_review_per_product['Average Review Score'].idxmax()]

    worst_product = avg_review_per_product.loc[avg_review_per_product['Average Review Score'].idxmin()]

    products = [best_product['Product Category'], worst_product['Product Category']]
    average_scores = [best_product['Average Review Score'], worst_product['Average Review Score']]
    fig = go.Figure(data=[
        go.Bar(name='Average Review Score', x=products, y=average_scores,
               marker_color=['green', 'red'])
    ])

    fig.update_layout(
        title='Best and Worst Product Categories Based on Average Review Score',
        xaxis_title='Product Category',
        yaxis_title='Average Review Score',
        yaxis=dict(range=[0, 5]),
        showlegend=False
    )

    st.plotly_chart(fig)
def subscription_counts():
    subscription_counts = df['Subscription Status'].value_counts()

    fig = px.pie(subscription_counts,
                 values=subscription_counts.values,
                 names=subscription_counts.index,
                 title='Subscription Status Percentage',
                 color_discrete_sequence=px.colors.qualitative.Pastel)

    st.title('Subscription Status Visualization')
    st.plotly_chart(fig)
def plot_age_distribution_premium():
    premium_customers = df[df['Subscription Status'] == 'Premium']
    bins = [0, 18, 35, 50, 65, 80]
    labels = ['0-17', '18-34', '35-49', '50-64', '65+']
    premium_customers['Age Range'] = pd.cut(premium_customers['Age'], bins=bins, labels=labels, right=False)
    age_range_counts = premium_customers['Age Range'].value_counts().reset_index(name='Count')
    age_range_counts.columns = ['Age Range', 'Count']
    top_age_ranges = age_range_counts.sort_values(by='Count', ascending=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x='Age Range', y='Count', data=top_age_ranges, palette='Blues_d', ax=ax)
    ax.set_title('Age Ranges among Premium Customers')
    st.pyplot(fig)

def plot_device_distribution():
    def create_donut_chart(subscription_status):
        customers = df[df['Subscription Status'] == subscription_status]
        device_counts = customers['Device Type'].value_counts().reset_index()
        device_counts.columns = ['Device Type', 'Count']
        return device_counts

    fig = make_subplots(rows=1, cols=3, specs=[[{'type': 'pie'}, {'type': 'pie'}, {'type': 'pie'}]],
                        subplot_titles=('Premium Subscribers', 'Free Subscribers', 'Trial Subscribers'))

    premium_data = create_donut_chart('Premium')
    free_data = create_donut_chart('Free')
    trial_data = create_donut_chart('Trial')

    fig.add_trace(go.Pie(values=premium_data['Count'], labels=premium_data['Device Type'], hole=0.4, name='Premium'),
                  row=1, col=1)
    fig.add_trace(go.Pie(values=free_data['Count'], labels=free_data['Device Type'], hole=0.4, name='Free'),
                  row=1, col=2)
    fig.add_trace(go.Pie(values=trial_data['Count'], labels=trial_data['Device Type'], hole=0.4, name='Trial'),
                  row=1, col=3)

    fig.update_layout(title_text='Device Distribution by Subscription Status', title_x=0.5)
    st.plotly_chart(fig)

def plot_purchases_and_delivery_time():
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    purchases_by_location = df.groupby('Location')['Purchase Amount ($)'].sum()
    purchases_by_location.plot(kind='bar', color='skyblue', ax=axs[0])
    axs[0].set_title('Purchases by Location')
    axs[0].set_xlabel('Location')
    axs[0].set_ylabel('Total Purchase Amount ($)')
    axs[0].tick_params(axis='x', rotation=45)

    avg_delivery_time_by_location = df.groupby('Location')['Delivery Time (days)'].mean()
    avg_delivery_time_by_location.plot(kind='bar', color='salmon', ax=axs[1])
    axs[1].set_title('Average Delivery Time by Location')
    axs[1].set_xlabel('Location')
    axs[1].set_ylabel('Average Delivery Time (days)')
    axs[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    st.pyplot(fig)
def device_users():


    devices = df['Device Type'].value_counts(normalize=True) * 100
    devices_df = devices.reset_index()
    devices_df.columns = ['Device Type', 'Percentage']

    fig = px.bar(devices_df,
                 x='Device Type',
                 y='Percentage',
                 title='Device Type Distribution',
                 labels={'Percentage': 'Percentage (%)'},
                 color='Percentage',
                 color_continuous_scale=px.colors.sequential.Viridis)

    st.title('Device Type Distribution Visualization')
    st.plotly_chart(fig)
def payment_method():
    payment_method_counts = df['Payment Method'].value_counts()

    fig = px.bar(payment_method_counts,
                 x=payment_method_counts.index,
                 y=payment_method_counts.values,
                 title='Distribution of Payment Methods',
                 labels={'x': 'Payment Method', 'y': 'Number of Customers'},
                 color=payment_method_counts.index)

    st.title('Payment Method Distribution Visualization')
    st.plotly_chart(fig)
def avg_review_payment():
    avg_review_by_payment = df.groupby('Payment Method')['Review Score (1-5)'].mean().reset_index()
    avg_review_by_payment.columns = ['Payment Method', 'Average Review Score']

    common_payment_method = df['Payment Method'].value_counts().idxmax()
    common_payment_users = df[df['Payment Method'] == common_payment_method]
    avg_review_cpu = common_payment_users['Review Score (1-5)'].mean()

    st.write(
        f'Average review score of users of the most common payment method ({common_payment_method}): {avg_review_cpu:.2f}')

    fig = px.bar(avg_review_by_payment,
                 x='Payment Method',
                 y='Average Review Score',
                 title='Average Review Score by Payment Method',
                 labels={'Average Review Score': 'Average Review Score'},
                 color='Average Review Score',
                 color_continuous_scale=px.colors.sequential.Viridis)

    st.title('Average Review Score by Payment Method')
    st.plotly_chart(fig)
def corr_time():
    corr_timespent_purchase = df['Time Spent on Website (min)'].corr(df['Purchase Amount ($)'])
    print(corr_timespent_purchase)
    fig = px.scatter(df,
                     x='Time Spent on Website (min)',
                     y='Purchase Amount ($)',
                     title='Correlation between Time Spent on Website and Purchase Amount',
                     labels={'Time Spent on Website (min)': 'Time Spent on Website (minutes)',
                             'Purchase Amount ($)': 'Purchase Amount ($)'})

    st.title('Correlation Analysis')
    st.plotly_chart(fig)
def satisfied_return():
    satisfied_customers = df[df['Review Score (1-5)'] >= 4]
    return_satisfied_customers = satisfied_customers[satisfied_customers['Return Customer'] == True]
    percentage_rsc = (len(return_satisfied_customers) / len(df)) * 100

    print(f"Percentage of satisfied return customers: {percentage_rsc:.2f}%")
    satisfied_customers = df[df['Review Score (1-5)'] >= 4]
    dissatisfied_customers = df[df['Review Score (1-5)'] <= 3]

    satisfied_return_percentage = (len(satisfied_customers[satisfied_customers['Return Customer'] == True]) / len(
        satisfied_customers)) * 100
    dissatisfied_return_percentage = (len(
        dissatisfied_customers[dissatisfied_customers['Return Customer'] == True]) / len(dissatisfied_customers)) * 100


    labels = ['Satisfied Return Customers', 'Satisfied Non-Return Customers',
              'Dissatisfied Return Customers', 'Dissatisfied Non-Return Customers']
    values = [
        len(satisfied_customers[satisfied_customers['Return Customer'] == True]),
        len(satisfied_customers[satisfied_customers['Return Customer'] == False]),
        len(dissatisfied_customers[dissatisfied_customers['Return Customer'] == True]),
        len(dissatisfied_customers[dissatisfied_customers['Return Customer'] == False])
    ]

    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
    fig.update_layout(title_text="Return Customer Percentage by Rating")


    st.title('Customer Satisfaction Analysis')
    st.plotly_chart(fig)
def corr_purchased_satisfaction():
    corr_itembuy_satisfaction = df['Number of Items Purchased'].corr(df['Review Score (1-5)'])
    print(corr_itembuy_satisfaction)
    fig = px.scatter(df,
                     x='Number of Items Purchased',
                     y='Review Score (1-5)',
                     title='Number of Items Purchased vs. Customer Satisfaction',
                     labels={'Number of Items Purchased': 'Number of Items Purchased',
                             'Review Score (1-5)': 'Satisfaction Review'},
                     color_discrete_sequence=['blue'])

    fig.update_xaxes(tickvals=list(range(0, 11)))
    fig.update_yaxes(tickvals=list(range(0, 7)))

    st.title('Customer Satisfaction Analysis')
    st.plotly_chart(fig)
def  highest_average():
    location_avg_purchase = df.groupby('Location')['Purchase Amount ($)'].mean()

    sorted_avg_purchase = location_avg_purchase.sort_values(ascending=False)

    second_highest_avg_purchase = sorted_avg_purchase.iloc[1]
    second_highest_location = sorted_avg_purchase.index[1]
    print(f"The location with the second highest average purchase amount is: {second_highest_location}")
    print(f"The second highest average purchase amount is: {second_highest_avg_purchase:.2f}")
    location_avg_purchase = df.groupby('Location')['Purchase Amount ($)'].mean().sort_values(ascending=False)
    fig = px.bar(location_avg_purchase,
                 x=location_avg_purchase.index,
                 y=location_avg_purchase.values,
                 title='Average Purchase Amount by Location (Sorted)',
                 labels={'x': 'Location', 'y': 'Average Purchase Amount'},
                 color=location_avg_purchase.values)


    st.title('Average Purchase Amount by Location')
    st.plotly_chart(fig)
def  factors_contribute():
    return_customer_purchase_category = df[df['Return Customer'] == True].groupby('Product Category')[
        'Customer ID'].count()

    return_customer_purchase_category_sorted = return_customer_purchase_category.sort_values(ascending=False)
    fig = px.bar(
        x=return_customer_purchase_category_sorted.index,
        y=return_customer_purchase_category_sorted.values,
        title='Return Customer Purchase Category',
        labels={'x': 'Product Category', 'y': 'Number of Return Customers'},
        color=return_customer_purchase_category_sorted.index
    )

    fig.update_layout(xaxis_tickangle=-45)


    st.title('Return Customer Purchase Category Analysis')
    st.plotly_chart(fig)
def payment_satisfaction(df):
    total_pay_method = df['Payment Method'].value_counts()
    total_pay_method = total_pay_method.sort_values(ascending=False)
    print(total_pay_method)
    satisfaction_by_payment = df.groupby(['Payment Method', 'Customer Satisfaction']).size().unstack(fill_value=0)
    satisfaction_by_payment = satisfaction_by_payment.sort_values(by='High', ascending=False)
    print("Customer Satisfaction by Payment Method:")
    print(satisfaction_by_payment)
    satisfaction_by_payment = df.groupby(['Payment Method', 'Customer Satisfaction'])['Customer ID'].count().unstack()

    satisfaction_percentage = satisfaction_by_payment.div(satisfaction_by_payment.sum(axis=1), axis=0) * 100
    fig_satisfaction = px.bar(
        satisfaction_percentage.reset_index(),
        x="Payment Method",
        y=["High", "Medium", "Low"],
        title="Customer Satisfaction by Payment Method",
        labels={"value": "Percentage of Customers", "variable": "Customer Satisfaction"},
        barmode="group"
    )

    st.title('Customer Satisfaction Analysis')
    st.plotly_chart(fig_satisfaction)

    return_data = {
        'Payment Method': ['Credit Card', 'PayPal', 'Debit Card'],
        'Return Customer': [0.8, 0.6, 0.4]
    }
    df = pd.DataFrame(return_data)

    return_rate_by_payment = df.groupby('Payment Method')['Return Customer'].mean() * 100


    fig_return_rate = px.bar(
        return_rate_by_payment.reset_index(),
        x="Payment Method",
        y="Return Customer",
        title="Return Rate by Payment Method",
        labels={"Return Customer": "Return Rate (%)"}
    )


    st.plotly_chart(fig_return_rate)
def location_influence():
    purchases_by_location = df.groupby('Location')['Purchase Amount ($)'].sum().reset_index()


    fig_purchases = px.bar(
        purchases_by_location,
        x='Location',
        y='Purchase Amount ($)',
        title='Purchases by Location',
        labels={'Purchase Amount ($)': 'Total Purchase Amount'},
        color='Purchase Amount ($)',
        color_continuous_scale='Blues'
    )


    avg_delivery_time_by_location = df.groupby('Location')['Delivery Time (days)'].mean().reset_index()


    fig_delivery_time = px.bar(
        avg_delivery_time_by_location,
        x='Location',
        y='Delivery Time (days)',
        title='Average Delivery Time by Location',
        labels={'Delivery Time (days)': 'Average Delivery Time (days)'},
        color='Delivery Time (days)',
        color_continuous_scale='Reds'
    )


    st.title('Purchases and Delivery Time Analysis')


    st.plotly_chart(fig_purchases)


    st.plotly_chart(fig_delivery_time)

def review():
    avg_review_by_payment = df.groupby('Payment Method')['Review Score (1-5)'].mean().reset_index()

    avg_review_by_payment.columns = ['Payment Method', 'Average Review Score']

    common_payment_method = df['Payment Method'].value_counts().idxmax()
    common_payment_users = df[df['Payment Method'] == common_payment_method]
    avg_review_cpu = common_payment_users['Review Score (1-5)'].mean()

    print(
        f'Average review score of users of the most common payment method ({common_payment_method}): {avg_review_cpu:.2f}')
    fig = px.bar(
        avg_review_by_payment,
        x='Payment Method',
        y='Average Review Score',
        title='Average Review Score by Payment Method',
        labels={'Average Review Score': 'Average Review Score'},
        color='Average Review Score',
        color_continuous_scale=px.colors.sequential.Viridis
    )

    # Display the chart in Streamlit
    st.title('Average Review Score Analysis')
    st.plotly_chart(fig)
def main():
    st.sidebar.title("Navigation")
    level = st.sidebar.radio("Choose Level", ['Level 1', 'Level 2', 'Level 3'])

    if level == 'Level 1':
        st.title("Level 1: Visualizations")
        option = st.selectbox("Choose a visualization", ["Mean, Median and Mode", "Variance, Standard Deviation, and Z-Score of Purchase Amount.", "The top Three Products.","Return Customers", "Average Review Score","total subscription status","Device users","Payment method used by customers", "Age Distribution (Premium Customers)", "Device Distribution by Subscription Status"])

        if option == "Mean, Median and Mode":
            st.subheader("Mean, Median and Mode")
            mean_median_mode()
            st.write("Here is Mean, Median and Mode.")

        elif option == "Variance, Standard Deviation, and Z-Score of Purchase Amount.":
            st.subheader("Variance, Standard Deviation, and Z-Score of Purchase Amount.")
            variance()
            st.write("Above we can easily understand the Variance, Standard Deviation, and Z-Score of Purchase Amount.")

        elif option == "The top Three Products.":
            st.subheader("The top Three Products.")
            top_products(df)
            st.write("")

        elif option == "Return Customers":
            st.subheader("Return Customers")
            return_customer()
            st.write("")

        elif option == "Average Review Score":
            st.subheader("Average Review Score")
            review_score()
            st.write("")
        elif option == "total subscription status":
            st.subheader("total subscription status")
            subscription_counts()
            st.write("This chart displays the distribution of age ranges for premium customers.")
        elif option == "Device users":
            st.subheader("Device users")
            device_users()
            st.write("")
        elif option == "Payment method used by customers":
            st.subheader("Payment method used by customers")
            payment_method()
            st.write("")

        elif option == "Device Distribution by Subscription Status":
            st.subheader("Device Distribution by Subscription Status")
            plot_device_distribution()
            st.write("This chart shows the distribution of devices used by premium, free, and trial subscribers.")

        elif option == "Age Distribution (Premium Customers)":
            st.subheader("Age Distribution for Premium Customers")
            plot_age_distribution_premium()
            st.write("This chart displays the distribution of age ranges for premium customers.")




    elif level == 'Level 2':
        st.title("Level 2: ")
        option = st.selectbox("Choose a visualization", ["The average review scores of users of the most common payment method","corretaion between time and purchese","Percentage of customers are satisfied (rating of 4 or 5) and are also return customers","The relationship between the number of items purchased and customer satisfaction","Location has the 2nd highest average purchase amount"])

        if option == "The average review scores of users of the most common payment method":
            st.subheader("The average review scores of users of the most common payment method")
            review()
            st.write("This chart shows the total purchases by location and the average delivery time by location.")
        elif option =="corretaion between time and purchese":
            st.subheader("corretaion between time and purchese")
            corr_time()
            st.write("")
        elif option =="Percentage of customers are satisfied (rating of 4 or 5) and are also return customers":
            st.subheader("Percentage of customers are satisfied (rating of 4 or 5) and are also return customers")
            satisfied_return()
            st.write("")
        elif option =="The relationship between the number of items purchased and customer satisfaction":
            st.subheader("The relationship between number of items purchased and customer satisfaction")
            corr_purchased_satisfaction()
            st.write("")
        elif option =="Location has the 2nd highest average purchase amount":
            st.subheader("Location has the 2nd highest average purchase amount.")
            highest_average()
            st.write("")

    elif level == 'Level 3':
        st.title("Level 3: ")
        option = st.selectbox("Choose a visualization", ["Payment methods influence customer satisfaction and return rates","Factors that influence customers return","The location influence both purchase amount and delivery time"])

        if option == "Factors that influence customers return":
            st.subheader("Factors that influence customers return")
            factors_contribute()
            st.write("")
        elif option == "Payment methods influence customer satisfaction and return rates":
            st.subheader("payment methods influence customer satisfaction and return rates")
            payment_satisfaction(df)
            st.write("")
        elif option == "The location influence both purchase amount and delivery time":
            st.subheader("the location influence both purchase amount and delivery time")
            location_influence()
            st.write("")


    elif level == 'My assumptions':
        st.title("Level 3: Coming Soon")
        st.write("Additional visualizations will be added here.")

if __name__ == "__main__":
    main()
