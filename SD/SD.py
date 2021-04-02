# function to youtube play videos in Chrome 
def open_in_chrome(url,path = "C://Program Files (x86)//Google//Chrome//Application//chrome.exe"):
    # takes youtube video ID
    # display it using chrome 
    # you might need to change the path to google chrome below
    import webbrowser
    webbrowser.register(
        'chrome',
        None,
        webbrowser.BackgroundBrowser(path)
        )
    webbrowser.get('chrome').open(url)

url = "https://www.youtube.com/watch?v=YX0kCReIZzk&ab_channel=SuneLehmann"
open_in_chrome(url)



# 