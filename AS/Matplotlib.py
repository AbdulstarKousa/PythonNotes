"""
@author: Abdulstar Kousa 
Matplotlib Notes 
"""

#################################
# 0. library: 
#################################
from matplotlib import pyplot as plt


#################################
# 1. CurvePlot:
#################################

# Cordenates:
x1 = [ls]
y1 = [ls]   

# figure size
plt.figure(figsize=(8, 8))

# Plot command:
plt.plot(x1,y1,'abbrToColor',label="string",linewidth=nr)

# same graph
x2 = [ls]
y2 = [ls]
plt.plot(x2,y2,'abbrToColor',label="string",linewidth=nr)    
     
# abbrToColor:
"""
green   =   'g' 
blue    =   'c' 
black   =   'k' 
pink    =   'm' 
red     =   'r'
or
color   =   'tab:gray'
"""
# Shaded for error
plt.fill_between(xs, ys-std_, ys+std_)  

# vertical line
plt.plot([x_fix, x_fix], [y_low,y_high], c="black")

# horizontal line
plt.plot([x_left, x_right], [y_fix,y_fix], c="black")

# line 
plt.plot([x1,x2], [y1,y2], c="black")


# write inside the figure:
plt.annotate(s="$\mu$",xy=(x_where + 0.1 , y_where + 0.1), size=20)

# title
plt.title("$Latex$ string")

# curve and axis info:
plt.ylabel("string")
plt.xlabel("string")

# ticks
plt.xticks(x, str_lst,rotation='vertical')
plt.yticks(y, str_lst,rotation='vertical')
"""
ticks = np.arange(0,len(Data.select_dtypes(exclude=object).columns),1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(['Zip', 'LAT.', 'LONG.', 'Injured','killed', 'Response', 'Year', 'Month', 'Day', 'Hour','Minute', 'S.L.M.', 'Pre.', 'S.F.','S. D.', 'F.S.H.', 
"""



# log(y)
plt.yscale('log')

# show label    
plt.legend()                                                 

# Squares
plt.grid(True,color='k')

# Show the plot
plt.show()                                                       


#################################
# 2. scatterPlot: 
#################################
#cordenates:
x1 = [ls]
y1 = [ls]   
    
# plot command:
plt.scatter(x,y,alpha=0.1,color='abbrToColor',marker="abbrToMarker")    
           

#################################
# 3. BarGraph: 
#################################
# cordenates:
x = [ls]
y = [ls]   
    
# plot command:
plt.bar(x,y)


#################################
# 4. hist:
#################################
L = [lst]
plt.hist(L, bins=100, density=True)


#################################
# 5. multiple plots:
#################################
fig, axs = plt.subplots(#nr_row, #nr_col, figsize=(15, 20), sharex=True)
fig.suptitle('super-title', fontsize=20)
for i,ax in enumerate(axs.flat):
    ax.set(title='sub-title')
    plt.plot(x[i],x[i],'abbrToColor',label="string",linewidth=nr,ax=ax)

# more function for the subplots (ax in the for loop above) 
axs[i].text(-0.3, y[i].max()*1.1, 'crime', fontsize=16) #writing in the subplot
axs[i].set_ylim(0, counts.max()*1.4)
axs[i].set_ylabel('Crime count')
axs[i].set_xlabel('Weekdays')
axs[i].set_xticklabels(counts.index, rotation=-10)




#################################
# 6. Style:
#################################
from matplotlib import style
style.use("ggplot")                                              


#################################
# 7. Jitter plot
#################################
import seaborn as sns # for jitter plot
plt.figure(figsize=(8, 8))
sns.stripplot(data_WL['Time'].values, jitter=True, edgecolor='none', alpha=.50 ,color='k').set_title('Robbery\nJan to Jul\n13:00-14:00')
plt.show()

