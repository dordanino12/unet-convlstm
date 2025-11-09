to use udi ernder i need to use for example : 
pkl_file = "/wdata_visl/udigal/samples/samples_mode3_res128_stride64_spp8/samples_3D/BOMEX_512x512x200_20m_20m_1s_512_0000002000_0_3.pkl"
from this file take the beta value , the dim of the filed is z,x,y with some padding and crop to the clouds area 

then use write_vol_file def in render_from_udi to build the vol file from this pkl file

i build run_render.py that still have issues , need undarstand the world there , the sun i white acoring udi 