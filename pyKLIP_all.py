import os
import spaceKLIP

# Run pyKLIP pipeline. Additional parameters for klip_dataset function can
# be passed using kwargs parameter.
os.environ["OPENBLAS_NUM_THREADS"] = "1"

if __name__ == "__main__":
        data_root = '../Data/F410M_LIKELY_Th8'
        input_subdir = 'coadded'
        output_subdir = 'klipsub_all'
        pid = 4758
        database = spaceKLIP.database.create_database(
                                        input_dir=os.path.join(data_root, input_subdir),
                                        file_type='calints.fits',
                                        output_dir=data_root,
                                        pid=pid)
        spaceKLIP.pyklippipeline.run_obs(database=database,
                        kwargs={'mode': ['RDI', 'ADI + RDI'],
                                'annuli': [3],
                                'subsections': [3],
                                'numbasis': [50],
                                'algo': 'klip',
                                'save_rolls': False},
                        subdir=output_subdir)
