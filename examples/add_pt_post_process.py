import pickle

from mosstool.map.public_transport.public_transport_post import \
    public_transport_process
from mosstool.type import Map
from mosstool.util.format_converter import dict2pb

# pre-route
# run: ./routing -map ./data/temp/srt.raw_pt_map.pb
LISTENING_HOST = "http://localhost:52101"
MAP_PKL_PATH = "./data/temp/srt.raw_pt_map.pkl"
PT_MAP_PATH = f"./data/temp/srt.map_with_pt.pb"
m_dict = pickle.load(open(MAP_PKL_PATH, "rb"))
new_m = public_transport_process(m_dict, LISTENING_HOST)
pb = dict2pb(new_m, Map())
with open(PT_MAP_PATH, "wb") as f:
    f.write(pb.SerializeToString())
