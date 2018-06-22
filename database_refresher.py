import os
import db_connection
from mmdbs_image import MMDBSImage
import datetime

print("Start time: ", datetime.datetime.now())

# Set environment variables
working_dir = os.path.dirname(os.path.realpath(__file__))
path = working_dir + "/source/"
print("Working directory is ", working_dir + ".")
print("The images are stored in ", path, ".")

# Establish DB connection
conn = db_connection.connect()

# Loop trough all subdirectories of the given path if refresh = TRUE
# The name of each subdirectory represents the class of the images inside

# Clean up database.
print("Clean up database.")
#number_of_deleted_images = db_connection.delete_all_images(conn)
#print(str(number_of_deleted_images), " images deleted in database.")

for subdirectory in os.listdir(path):

    # Build absolute subdirectory path
    subdirectory_path = (path + subdirectory)

    # Ignore files for the MacOS file system
    if subdirectory != ".DS_Store":
        for image in os.listdir(subdirectory_path):
            if image != ".DS_Store":
                start_time_of_image = datetime.datetime.now()

                image_path = subdirectory_path + "/" + image

                print("Read image ", image_path)

                # Build image object
                temp_image = MMDBSImage()
                temp_image.set_image(image_path, subdirectory)
                temp_image.extract_features('all')

                # See outputs
                # cv2.imwrite(image, temp_image.sobel_edge_detection)
                # print(temp_image.global_edge_histogram)

                # Write the features of the Image object to the database
                db_connection.write_image_to_database(conn, temp_image)

                end_time_of_image = datetime.datetime.now()

                computation_time = end_time_of_image-start_time_of_image

                print("Computation took ", computation_time.seconds, ".", computation_time.microseconds," seconds.")
                print("===")

# Close database connection
db_connection.close_connection(conn)

print("End time: ", datetime.datetime.now())

