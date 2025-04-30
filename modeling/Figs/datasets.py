import pandas as pd

# Reconstruct the reordered data again after reset
data = {
    'DatasetName': ['msg_bt_f64', 'num_brain_f64', 'num_control_f64', 'rsim_f32', 'astro_mhd_f64',
                    'astro_pt_f64', 'turbulence_f32', 'wave_f32', 'citytemp_f32', 'wesad_chest_f64',
                    'solar_wind_f32', 'acs_wht_f32', 'hdr_night_f32', 'hdr_palermo_f32',
                    'hst_wfc3_uvis_f32', 'hst_wfc3_ir_f32', 'spitzer_irac_f32', 'g24_78_usb2_f32',
                    'jw_mirimage_f32', 'tpch_order_f64', 'tpcxbb_store_f64', 'tpcxbb_web_f64',
                    'tpcds_catalog_f32', 'tpcds_store_f32', 'tpcds_web_f32'],
    'Domain': ['HPC', 'HPC', 'HPC', 'HPC', 'HPC', 'HPC', 'HPC', 'HPC', 'TS', 'TS',
               'TS', 'OBS', 'OBS', 'OBS', 'OBS', 'OBS', 'OBS', 'OBS', 'OBS',
               'DB', 'DB', 'DB', 'DB', 'DB', 'DB'],
    'Entropy': [7.565735873, 7.502112859, 7.785477575, 6.318406203, 1.481182889,
                7.651160363, 7.271298432, 7.551788764, 4.810124503, 4.645564788,
                6.771888678, 6.442601304, 3.567375417, 3.882997527, 5.467166095,
                5.410832718, 6.906106101, None, 7.368734063, 6.706198226,
                6.025990953, 6.110776173, 6.760220463, 6.345535468, 6.762170614]
}

df_reordered = pd.DataFrame(data)
df_reordered.insert(0, 'DatasetID', ['D{}'.format(i+1) for i in range(len(df_reordered))])




# Save
output_csv_path = '/mnt/c/Users/jamalids/Downloads/reordered_datasets.csv'
df_reordered.to_csv(output_csv_path, index=False)

output_csv_path
