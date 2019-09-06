# importing the library
from google_images_download import google_images_download

# setting arguments
# supporting multiple keywords, e.g. "Keyword1,Keyword2,Keyword3"
KEYWORDS = "Hosier Lane"
LIMIT_NUM = 200  # per keyword
OUTPUT_PATH = r"E:\landmarks_of_melbourne"
CHROMEDRIVER_PATH = r"C:\Users\PC-user\Downloads\chromedriver.exe"
CHROMEDRIVER_PATH = r"C:\Users\Administrator\Downloads\chromedriver.exe"

# class instantiation
response = google_images_download.googleimagesdownload()

# creating list of arguments
arguments = {"keywords": KEYWORDS,
             "limit": LIMIT_NUM,
             "print_urls": True,
             "output_directory": OUTPUT_PATH,
             "size": ">400*300",
             "chromedriver": CHROMEDRIVER_PATH
             }

# passing the arguments to the function
paths = response.download(arguments)

# printing absolute paths of the downloaded images
# print(paths)
