# importing the library
from google_images_download import google_images_download

# setting arguments
LIMIT_NUM = 600
KEYWORDS = "Immigration Museum Melbourne"
OUTPUT_PATH = "E:\\landmarks_of_melbourne"

# class instantiation
response = google_images_download.googleimagesdownload()

# creating list of arguments
arguments = {"keywords": KEYWORDS,
             "limit": LIMIT_NUM,
             "print_urls": True,
             "output_directory": OUTPUT_PATH,
             "size": ">400*300",
             "chromedriver": r"C:\Users\PC-user\Downloads\chromedriver.exe"}

# passing the arguments to the function
paths = response.download(arguments)

# printing absolute paths of the downloaded images
# print(paths)
