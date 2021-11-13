# """
# @author: Abdulstar Kousa 
# Matplotlib Notes 
# """
# #################################
# # 0. library: 
# #################################
# from operator import imod
# import IPython
# from IPython.lib.display import YouTubeVideo
# from matplotlib import pyplot as plt, style


# #################################
# # 1. CurvePlot:
# #################################

# # ax
# plt.gca()

# # Cordenates:
# x1 = [ls]
# y1 = [ls]   

# # figure size
# plt.figure(figsize=(8, 8))

# # Plot command:
# plt.plot(x1,y1,'abbrToColor',label="string",linewidth=nr)

# # same graph
# x2 = [ls]
# y2 = [ls]
# plt.plot(x2,y2,'abbrToColor',label="string",linewidth=nr)    
     
# # abbrToColor:
# """
# green   =   'g' 
# blue    =   'c' 
# black   =   'k' 
# pink    =   'm' 
# red     =   'r'
# or
# color   =   'tab:gray'
# """
# # Shaded for error
# plt.fill_between(xs, ys-std_, ys+std_)  

# # vertical line
# plt.plot([x_fix, x_fix], [y_low,y_high], c="black")

# # horizontal line
# plt.plot([x_left, x_right], [y_fix,y_fix], c="black")

# # line 
# plt.plot([x1,x2], [y1,y2], c="black")


# # write inside the figure:
# plt.annotate(s="$\mu$",xy=(x_where + 0.1 , y_where + 0.1), size=20)

# # title
# plt.title("$Latex$ string")

# # curve and axis info:
# plt.ylabel("string")
# plt.xlabel("string")

# # ticks
# plt.xticks(ticks=bins, labels=labels)
# plt.xticks(x, str_lst,rotation='vertical')
# plt.yticks(y, str_lst,rotation='vertical')
# """
# ticks = np.arange(0,len(Data.select_dtypes(exclude=object).columns),1)
# ax.set_xticks(ticks)
# ax.set_yticks(ticks)
# ax.set_xticklabels(['Zip', 'LAT.', 'LONG.', 'Injured','killed', 'Response', 'Year', 'Month', 'Day', 'Hour','Minute', 'S.L.M.', 'Pre.', 'S.F.','S. D.', 'F.S.H.', 
# """



# # log(y)
# plt.yscale('log')

# # show label    
# plt.legend()                                                 

# # Squares
# plt.grid(True,color='k')

# # Show the plot
# plt.show()                                                       


# #################################
# # 2. scatterPlot: 
# #################################
# #cordenates:
# x1 = [ls]
# y1 = [ls]   
    
# # plot command:
# plt.scatter(x,y,alpha=0.1,color='abbrToColor',marker="abbrToMarker")    
           

# #################################
# # 3. BarGraph: 
# #################################
# # cordenates:
# x = [ls]
# y = [ls]   
    
# # plot command:
# plt.bar(x,y)


# #################################
# # 4. hist:
# #################################
# L = [lst]
# plt.hist(L, bins=100, density=True)


# #################################
# # 5. multiple plots:
# #################################
# fig, axs = plt.subplots(#nr_row, #nr_col, figsize=(15, 20), sharex=True)
# fig.suptitle('super-title', fontsize=20)
# for i,ax in enumerate(axs.flat):
#     ax.set(title='sub-title')
#     plt.plot(x[i],x[i],'abbrToColor',label="string",linewidth=nr,ax=ax)

# # more function for the subplots (ax in the for loop above) 
# axs[i].text(-0.3, y[i].max()*1.1, 'crime', fontsize=16) #writing in the subplot
# axs[i].set_ylim(0, counts.max()*1.4)
# axs[i].set_ylabel('Crime count')
# axs[i].set_xlabel('Weekdays')
# axs[i].set_xticklabels(counts.index, rotation=-10)


# enumerate 

# #################################
# # 6. Style:
# #################################
# from matplotlib import style
# style.use("ggplot")                                              


# #################################
# # 7. Jitter plot
# #################################
# import seaborn as sns # for jitter plot
# plt.figure(figsize=(8, 8))
# sns.stripplot(data_WL['Time'].values, jitter=True, edgecolor='none', alpha=.50 ,color='k').set_title('Robbery\nJan to Jul\n13:00-14:00')
# plt.show()


# #################################
# # Examples: 
# #################################

# """ 1 """
# plt.figure(figsize=(8,6)) # define the figure size 
# #plt.title('Title Needed') # Having a title is always needed for plots :) 
# plt.ylabel('$E_{1D} - E_{bulk}$  (eV/atom)')
# plt.xlabel('$s_{0} + s_{01} + s_{1}$') # The '$' signs allow to use latex formated string

# plt.xlim([0.5, 1]) # Set x-axis range 
# ax = plt.gca() # Take the plot needed to play with xticks 
# bins = np.arange(0.5, 1.05, 0.1) # Define the bins - Here we have as a max limit 1.05 to include 1 
# bins_labels = [str(np.round(i,1)) for i in bins] # Define the bins as list of string items - Here we do round to 1 dec to prevent having to many decimals in the plot due to machine error  
# ax.set_xticks(bins) # This is our xticks
# ax.set_xticklabels(bins_labels) # This is what we want to show instead of our xticks (Here by chance the replaced one and xticks are the same but in general this could be any list of string that have the same size as our xticks)

# inds  = np.digitize(x,bins) # Return the indices (place) of the bins to which each value from input array x belongs
# means = [np.mean(y[inds==i]) for i in list(set(inds))] # Calculate the means ask if needed 
# stds   = [np.std(y[inds==i], ddof=1) for i in list(set(inds))] # Calculate the std ask if needed
# delta = bins[1]-bins[0] # to center of each bin 
# mid_bins = bins - delta/2 # Center of each bin
# mid_bins = mid_bins[1:] # need to be deleted since it is less than the left bordar of the first bin

# plt.plot(x,y,'bo',markersize=2) # Plot x vs y values
# plt.errorbar(mid_bins, means, stds, linestyle='None', marker='o', color='r') # Plot mean and standard deviation
# plt.grid(True) # squares in the plot 
# plt.savefig('Title Needed') # save to same directory 
# plt.show() 


# """ 2 density """
# def Visual_Histogram(x, n_bins = 10):
#     bins = np.linspace(min(x), max(x), n_bins + 1)    
#     weights = np.ones(len(x))/len(x)
#     plt.hist(x, n_bins, weights=weights, ec='black') # ec short for "edgecolor"
#     plt.title("Density Histogram")
#     plt.xlabel("Classes")
#     plt.xticks(ticks=bins, labels=[str(xi) for xi in np.round(bins,2)])
#     plt.ylabel("Density")
#     plt.show()
# Visual_Histogram(x, n_bins = 10)



# def density_Histogram(x, n_bins = 10, figsize=(10,6), title= "Density Histogram", xlabel= 'Classes', ylabel='Density', xticks=False):
#     plt.figure(figsize=figsize)
#     n_bins = n_bins
#     bins = np.linspace(min(x), max(x), n_bins + 1)    
#     weights = np.ones(len(x))/len(x)
#     plt.hist(x, n_bins, weights=weights, ec='black') # ec short for "edgecolor"
#     plt.title(title)
#     plt.xlabel(xlabel)
#     if xticks : plt.xticks(ticks=bins, labels=[str(xi) for xi in np.round(bins,2)])
#     plt.ylabel(ylabel)
#     plt.show()


# from IPython import YouTubeVideo

# YouTubeVideo()


# """ more """
# import matplotlib as mpl
# mpl.rcParams.update(mpl.rcParamsDefault)

# from matplotlib.pyplot import style
# style.use('ggplot')

# %matplotlib inline

# from IPython.display import display
# from IPython import get_ipython
# get_ipython().run_line_magic('matplotlib', 'inline')


# ax = plt.gca()