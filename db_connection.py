# !/usr/bin/python
import json
import sys
from configparser import ConfigParser
import psycopg2

from mmdbs_image import MMDBSImage


def connect():
    """
    Create a connection to the PostgreSQL database.
    :return: Connection object
    """

    try:
        # read database configuration
        params = config()
        # connect to the PostgreSQL database
        conn = psycopg2.connect(**params)
        print("Connected to database.")
        return conn

    except (Exception, psycopg2.DatabaseError) as error:
        print("An error occurred in connect().")
        print(error)


def close_connection(conn):
    """
    Close database connection.
    :param conn: Connection object
    """
    conn.close()
    print("Disconnected from database.")


def config(filename='database.ini', section='postgresql'):
    """
    Read the .ini containing the database credentials and store them in a dictionary.
    :param filename: file name of the configuration .ini
    :param section: corresponding section of the .ini
    :return: Dictionary containing the database credentials
    """
    # Create a parser
    parser = ConfigParser()
    # Read config file
    parser.read(filename)

    # Get section
    db = {}
    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            db[param[0]] = param[1]
    else:
        raise Exception('Section {0} not found in the {1} file'.format(section, filename))

    return db


def write_image_to_database(conn, image):
    """
    Inserts the image's features to the database.
    :param conn: connection object to access the database
    :param image: image object, whose features need to be stored in the database
    """

    sql = "INSERT INTO mmdbs_image(" \
          "path, " \
          "classification, " \
          "local_histogram1, " \
          "local_histogram2, " \
          "local_histogram3, " \
          "global_histogram, " \
          "global_edge_histogram, " \
          "global_hue_histogram) " \
          "VALUES(%s, %s, %s, %s, %s, %s, %s, %s) "

    try:
        # Create a new cursor
        cur = conn.cursor()
        # Execute the INSERT statement
        cur.execute(sql,
                    (image.path,
                     image.classification,
                     json.dumps(image.local_histogram_2_2),
                     json.dumps(image.local_histogram_3_3),
                     json.dumps(image.local_histogram_4_4),
                     json.dumps(image.global_histogram),
                     json.dumps(image.global_edge_histogram),
                     json.dumps(image.global_hue_histogram),)
                    )
        # Commit the changes to the database
        conn.commit()

        print("Saved image " + image.path + " to database.")

        # Close communication with the database
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print("An error occurred in write_image_to_database().")
        print(error)
        print(sys.exc_info()[0])


def get_all_images(conn):
    sql = "SELECT path, " \
          "classification, " \
          "local_histogram1, " \
          "local_histogram2, " \
          "local_histogram3, " \
          "global_histogram, " \
          "global_edge_histogram, " \
          "global_hue_histogram " \
          "FROM mmdbs_image"

    MMDBS_images = []

    try:
        # Create a new cursor
        cur = conn.cursor()
        # Execute the SELECT statement
        cur.execute(sql)
        rows = cur.fetchall()
        print("Found images: ", cur.rowcount)
        for row in rows:
            temp_MMDBS_image = MMDBSImage()
            temp_MMDBS_image.path = row[0]
            temp_MMDBS_image.classification = row[1]
            temp_MMDBS_image.local_histogram_2_2 = row[2]
            temp_MMDBS_image.local_histogram_3_3 = row[3]
            temp_MMDBS_image.local_histogram_4_4 = row[4]
            temp_MMDBS_image.global_histogram = row[5]
            temp_MMDBS_image.global_edge_histogram = row[6]
            temp_MMDBS_image.global_hue_histogram = row[7]
            MMDBS_images.append(temp_MMDBS_image)
        # Close communication with the database
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print("An error occurred in get_all_images().")
        print(error)

    return MMDBS_images


def get_image(conn):
    sql = """SELECT path FROM mmdbs_image LIMIT 1"""

    try:
        # Create a new cursor
        cur = conn.cursor()
        # Execute the SELECT statement
        cur.execute(sql)
        # Close communication with the database
        cur.close()
        return cur.fetchall()
    except (Exception, psycopg2.DatabaseError) as error:
        print("An error occurred in get_image().")
        print(error)


def get_count_images(conn):
    sql = """SELECT count(id) FROM mmdbs_image"""

    try:
        # Create a new cursor
        cur = conn.cursor()
        # Execute the SELECT statement
        cur.execute(sql)
        return cur.fetchone()
    except (Exception, psycopg2.DatabaseError) as error:
        print("An error occurred in get_count_images().")
        print(error)


def delete_all_images(conn):
    """
    Clear the table mmdbs_image in the database".
    :param conn: Connection object to access the database.
    :return: The numbe of deleted images.
    """
    rows_deleted = 0
    try:
        # create a new cursor
        cur = conn.cursor()
        # execute the UPDATE  statement
        cur.execute("DELETE FROM mmdbs_image")
        # get the number of updated rows
        rows_deleted = cur.rowcount
        # Commit the changes to the database
        conn.commit()
        # Close communication with the PostgreSQL database
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print("An error occurred in delete_all_images().")
        print(error)

    return rows_deleted
