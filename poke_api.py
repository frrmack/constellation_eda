import requests
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
import pickle
import csv
from collections import defaultdict
from json.decoder import JSONDecodeError

API_HOST = "https://nwconstellation.azurewebsites.net"
SEARCH_URL = API_HOST + '/AlphaSearch'

EXAMPLE_QUERY = "https://nwconstellation.azurewebsites.net/AlphaSearch?q_type=name&q_string=dr-samit-shah-md-11361147"

# SEARCH PARAMETERS
# insurance
# latitude
# longitude
# q_string   
# q_type    { name | specialty }
# age
# gender
# condition


def query_api(**kwargs):
    response = requests.get(SEARCH_URL, params=kwargs)
    try:
        result = response.json()
    except JSONDecodeError:
        print("WARNING: JSON ERROR!")
        print(f"HERE IS THE RESPONSE:\n{response.text}")
        return 0, 0
    pprint({"results": result['counts']})
    doc_count = result['counts']['doctors']
    pat_count = result['counts']['patients']
    params = list(kwargs.keys())
    if len(params)== 5 and 'q_type' in params and kwargs['q_type'] == 'specialty':
        print(f"Warning --{params} specialty only search, similar patient count is equal to total patients")
        print(f"old count: {pat_count}")
        pat_count = np.sum([node['totalPatients'] for node in result['graphData']['nodes'] if node['type']=='Doctor'])
        print(f"new count: {pat_count}")
    if pat_count is None:
        pat_count = 0
    # histogram of similarPatient number distirubtion among doctors
    # pat_counts = [doc.get('similarPatients', 0) for doc in result['graphData']['nodes']]
    # plt.hist(pat_counts, density=False, bins=30, log=True)  
    # plt.show()
    return doc_count, pat_count



# query_api(latitude=40.7867, longitude=-73.727, insurance="wellcare-wellcare-access", condition="vitamin-b12-deficiency")
# query_api(latitude=40.7867, longitude=-73.727, insurance="wellcare-wellcare-access")



# counts = {}
# counts['condition'] = requests.get("http://localhost:5000/Conditions").json()['counts']
# counts['genders'] = requests.get("http://localhost:5000/Genders").json()['counts']
# counts['ages'] = requests.get("http://localhost:5000/Ages").json()['counts']
# counts['specialties'] = requests.get("http://localhost:5000/Specialties").json()
# counts['zipcodes'] = requests.get("http://localhost:5000/Zipcodes").json()
# counts['insurances'] = requests.get("http://localhost:5000/Insurances").json()["counts"]
# with open("patient_distributions.pkl", "wb") as cachefile:
#     pickle.dump(counts, cachefile)

# # Load the counts
# with open("patient_distributions.pkl", "rb") as cachefile:
#     counts = pickle.load(cachefile)

# PROBS = {}
# for field, entry_count_dict in counts.items():
#     print(f'normalizing {field}')
#     n_all = np.sum(list(entry_count_dict.values()))
#     n_entries = len(entry_count_dict.items())
#     PROBS[field] = {entry: ((entry_count+1.0) / (n_all+n_entries)) for entry, entry_count in entry_count_dict.items() }

# PROBS['age'] = PROBS['ages']
# del PROBS['ages']
# PROBS['gender'] = PROBS['genders']
# del PROBS['genders']
# with open("patient_dist_probs.pkl", "wb") as cachefile:
#     pickle.dump(PROBS, cachefile)

# PROBS['specialty'] = PROBS['specialties']
# del PROBS['specialties']
# with open("patient_dist_probs.pkl", "wb") as cachefile:
#     pickle.dump(PROBS, cachefile)


# zip_to_lat_lon = {}
# with open("us-zip-code-latitude-and-longitude.csv", "r") as csvfile:
#      reader = csv.reader(csvfile, delimiter=';')
#      for row in reader:
#         zip, city, state, lat, lon, timezone, dst, geopoint = row
#         zip_to_lat_lon[zip] = (lat, lon)
# with open("zip_to_coord.pkl", "wb") as cachefile:
#     pickle.dump(zip_to_lat_lon, cachefile)



# Load zipcode translator
with open("zip_to_coord.pkl", "rb") as cachefile:
    ZIP_TO_COORD = pickle.load(cachefile)

ZIP_TO_COORD["10075"] = (40.77, -73.96)
ZIP_TO_COORD["10065"] = (40.765, -73.965)
ZIP_TO_COORD["10042"] = (40.704266, -74.006997)



# Load the probs
with open("patient_dist_probs.pkl", "rb") as cachefile:
    PROBS = pickle.load(cachefile)



def sample_field(field_name):
    sampled_x = np.random.choice(np.array(list(PROBS[field_name].keys())),
                                 p=np.array(list(PROBS[field_name].values())))
    if field_name == "zipcodes":
        return ZIP_TO_COORD[sampled_x]
    else:
        return sampled_x




ALL_FIELDS = list(PROBS.keys())
ALL_FIELDS.remove("insurances")
ALL_FIELDS.remove("zipcodes")

N_FIELDS_TO_COLOR = {
    0: "#fdeae1",
    1: "#fbb4b9",
    2: "#f768a1",
    3: "#c51b8a",
    4: "#7a0177",
}

def do_search(fields_list):
    lat, lon = sample_field("zipcodes")
    insurance = sample_field("insurances")
    fields_to_search_with = {}
    for field in fields_list:
        if field == "specialty":
            fields_to_search_with["q_type"] = "specialty"
            fields_to_search_with["q_string"] = sample_field("specialty")
        else:
            fields_to_search_with[field] = sample_field(field)
    pprint({"doctor fields": {'lat':lat, 'lon': lon, 'insurance': insurance}})
    pprint({"patient fields": fields_to_search_with})
    doc_c, pat_c =  query_api(latitude=lat,
                              longitude=lon,
                              insurance=insurance,
                              **fields_to_search_with)
    return lat, lon, insurance, fields_to_search_with, doc_c, pat_c


def random_search(n_patient_fields=None, fields_to_search_with=None):
    if n_patient_fields is None:
        n_patient_fields = np.random.choice([0,1,2,3,4])
    if fields_to_search_with is None:
        fields_to_search_with = np.random.choice(ALL_FIELDS, size=n_patient_fields, replace=False)
    else:
        n_patient_fields = len(fields_to_search_with)
    return n_patient_fields, do_search(fields_to_search_with)


def test_random_searches_for_fields(n_searches=10, n_patient_fields=None, fields_to_search_with=None, title_override=None):
    search_records = []
    doc_c, pat_c = [], []
    n_fields_to_doc_c, n_fields_to_pat_c = defaultdict(list), defaultdict(list)
    for i in range(n_searches):
        print(f"\nSEARCH {i+1}")
        n_patient_fields_for_this_search, (lat, lon, insurance,
                           search_fields, dc, pc) = random_search(fields_to_search_with=fields_to_search_with)
        # pepper with +-0.5 random noise to avoid plot points 
        # with same coords obscuring each other
        scattered_dc = dc + (1. * np.random.random()) - 0.5
        scattered_pc = pc + (1. * np.random.random()) - 0.5
        #
        doc_c.append(dc)
        pat_c.append(pc)
        n_fields_to_doc_c[n_patient_fields_for_this_search].append(scattered_dc)
        n_fields_to_pat_c[n_patient_fields_for_this_search].append(scattered_pc)
        search_records.append({
            "lat": lat,
            "lon": lon,
            "insurance": insurance,
            "condition": search_fields.get("condition", ""),
            "specialty": search_fields.get("q_string", ""),
            "age": search_fields.get("age", ""),
            "gender": search_fields.get("gender", ""),
            "doctor count": dc,
            "patient count": pc
        })
    # save results to a csv file
    results_file_name_slug = f"test_search_with_fields_{'_and_'.join(fields_to_search_with)}"
    with open(results_file_name_slug + ".csv", "w",  newline='') as csvfile:
        dict_writer = csv.DictWriter(csvfile, search_records[0].keys())
        dict_writer.writeheader()
        dict_writer.writerows(search_records)
    # pprint(n_fields_to_doc_c)
    # pprint(n_fields_to_pat_c)
    # plot results
    fig, ax = plt.subplots()
    for n in range(5):
        if n in n_fields_to_doc_c:
            ax.scatter(n_fields_to_doc_c[n],
                       n_fields_to_pat_c[n],
                       color=N_FIELDS_TO_COLOR[n],
                       label=f"{n} search fields",
                       edgecolors='none',
                       alpha=0.95)
    # ax.legend()
    ax.grid(True, color="#dcdcdc")
    ax.set_xlim(-1, 10*max(doc_c))
    ax.set_ylim(-1, 10*max(pat_c))
    ax.set_xlabel("doctor count")
    ax.set_ylabel("patient count")
    ax.set_yscale("symlog")
    ax.set_xscale("symlog")
    field_number_report = n_patient_fields
    if n_patient_fields is None:
        field_number_report = "a Random Number of"
    field_plural_or_singular = 'Fields'
    if len(fields_to_search_with) == 1:
        field_plural_or_singular = 'Field'
    default_title = f"Search Results with {field_plural_or_singular} {' + '.join([f.capitalize() for f in fields_to_search_with])}"
    if title_override is not None:        
        ax.set_title(title_override)
    else:
        ax.set_title(default_title)
    plt.savefig(results_file_name_slug + ".png")
    plt.show()



def test_random_searches(n_searches=10, n_patient_fields=None, title_override=None):
    search_records = []
    doc_c, pat_c = [], []
    n_fields_to_doc_c, n_fields_to_pat_c = defaultdict(list), defaultdict(list)
    for i in range(n_searches):
        print(f"\nSEARCH {i+1}")
        n_patient_fields_for_this_search, (lat, lon, insurance,
                           search_fields, dc, pc) = random_search(n_patient_fields=n_patient_fields)
        # pepper with +-0.5 random noise to avoid plot points 
        # with same coords obscuring each other
        scattered_dc = dc + (1. * np.random.random()) - 0.5
        scattered_pc = pc + (1. * np.random.random()) - 0.5
        #
        doc_c.append(dc)
        pat_c.append(pc)
        n_fields_to_doc_c[n_patient_fields_for_this_search].append(scattered_dc)
        n_fields_to_pat_c[n_patient_fields_for_this_search].append(scattered_pc)
        search_records.append({
            "lat": lat,
            "lon": lon,
            "insurance": insurance,
            "condition": search_fields.get("condition", ""),
            "specialty": search_fields.get("q_string", ""),
            "age": search_fields.get("age", ""),
            "gender": search_fields.get("gender", ""),
            "doctor count": dc,
            "patient count": pc
        })
    # save results to a csv file
    field_slug = n_patient_fields
    if field_slug is None: 
        field_slug = 'random'
    results_file_name_slug = f"test_search_{field_slug}_fields_{n_searches}_searches"
    with open(results_file_name_slug + ".csv", "w",  newline='') as csvfile:
        dict_writer = csv.DictWriter(csvfile, search_records[0].keys())
        dict_writer.writeheader()
        dict_writer.writerows(search_records)
    # pprint(n_fields_to_doc_c)
    # pprint(n_fields_to_pat_c)
    # plot results
    fig, ax = plt.subplots()
    for n in range(5):
        if n in n_fields_to_doc_c:
            ax.scatter(n_fields_to_doc_c[n],
                       n_fields_to_pat_c[n],
                       color=N_FIELDS_TO_COLOR[n],
                       label=f"{n} search fields",
                       edgecolors='none',
                       alpha=0.95)
    ax.legend()
    ax.grid(True, color="#dcdcdc")
    ax.set_xlim(-1, 10*max(doc_c))
    ax.set_ylim(-1, 10*max(pat_c))
    ax.set_xlabel("doctor count")
    ax.set_ylabel("patient count")
    ax.set_yscale("symlog")
    ax.set_xscale("symlog")
    field_number_report = n_patient_fields
    if n_patient_fields is None:
        field_number_report = "a Random Number of"
    default_title = f"Search Results with {field_number_report} Search Fields"
    if title_override is not None:        
        ax.set_title(title_override)
    else:
        ax.set_title(default_title)
    plt.savefig(results_file_name_slug + ".png")
    plt.show()


if __name__ == '__main__':
    # test_random_searches(n_searches=500,
    #                      n_patient_fields=1)

    
    # test_random_searches_for_fields(n_searches=500,
    #                                  fields_to_search_with=["specialty"])

    # test_random_searches_for_fields(n_searches=500,
    #                                  fields_to_search_with=["condition"])

    test_random_searches_for_fields(n_searches=500,
                                     fields_to_search_with=["specialty", "condition"])

