id_1 = "308353499"
id_2 = "308046994"
id_3 = "302893680"
email = "ofrik@mail.tau.ac.il"

def get_details():
    if (not id_1) or (not id_2) or not (email):
        raise Exception("Missing submitters info")

    info = str.format("{}_{}_{}      email: {}", id_1, id_2, id_3, email)

    return info