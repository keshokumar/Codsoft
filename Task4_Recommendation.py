import pandas as ps
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

users=['Virat','Kohli','Kesho','Kumar','Usha','Tamil','Deva','Rani','Arasi','Yamini',
      'Ajay','Robin','Dhanush','Aarav','Anaya','Diya','Vidya','Yash','Zara','Ishan']
movies=['Dangal', '3 Idiots', 'Gully Boy', 'Chhichhore', 'Bajrangi Bhaijaan',
        'Mersal', 'Kaala', 'Super Deluxe', '96', 'Kabali']

np.random.seed(42)
ratings=[]

for user in users:
    rate=np.random.choice(movies,size=5,replace=False)
    for movie in rate:
        ratings.append({'User':user.lower(),'Movie':movie,'Rating':np.random.randint(1,6)})
data=ps.DataFrame(ratings)

print("Sample data:")
print(data)

um_matrix=data.pivot(index='User',columns='Movie',values='Rating').fillna(0)

user_sim=cosine_similarity(um_matrix)
user_sim_data=ps.DataFrame(user_sim,index=um_matrix.index,columns=um_matrix.index)

def recommend_movies(user,num_rec):
    user=user.lower()
    if user not in um_matrix.index:
        return "User not found."
    user_rating=um_matrix.loc[user]
    sim_users=user_sim_data[user].sort_values(ascending=False)[1:]
    recommendation={}

    for sim_user in sim_users.index:
        sim_user_rating=um_matrix.loc[sim_user]
        for movie,rating in sim_user_rating.items():
            if user_rating[movie]==0 and movie not in recommendation:
                recommendation[movie]=rating
    
    recommend_movies=sorted(recommendation.items(),key=lambda x:x[1],reverse=True)[:num_rec]
    return recommend_movies

uname=input("Enter user name:")
num_rec=int(input("Enter the rating:"))

recommendation=recommend_movies(uname,num_rec)
print("\n Recommendation for {uname}:")
if isinstance(recommendation,str):
    print(recommendation)
else:
    for movie,rating in recommendation:
        print(f"{movie}:Predicted rating {rating:.2f}")
