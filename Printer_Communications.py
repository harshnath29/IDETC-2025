import requests
import json
import time


def issue_gcode(ip, com='', filename=""):

    base_request = ("http://{0}:{1}/rr_gcode?gcode=" + com).format(ip,"")
    r = requests.get(base_request)
    return r



def request_z_position(ip):

    base_request = ("http://{0}:{1}/rr_status?type=0").format(ip,"")
    return requests.get(base_request).json()['coords']['xyz'][2]

def request_printing_status(ip):

    base_request = ("http://{0}:{1}/rr_status?type=0").format(ip,"")
    return requests.get(base_request).json()


if __name__ == "__main__":

    ip1 = "192.168.0.18"
    print(request_printing_status(ip1)) # Need to clean up method to return True or False on if printer is printing
    print(request_z_position(ip1))

    issue_gcode(ip1, "M25")  # M25 = pause print; M24 = resume print

