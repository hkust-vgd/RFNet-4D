Data and scripts for Dynamic FAUST: Registering Human Bodies in Motion, CVPR 2017

==========

We supply raw scans, aligned templates (registrations) and masks of scan points with ground-truth accuracy.

The data is available for research purposes. Please check the license at
http://dfaust.is.tue.mpg.de

You can find the paper at
http://files.is.tue.mpg.de/black/papers/dfaust2017.pdf

To learn more about the project, please visit our website:
http://dfaust.is.tue.mpg.de

For comments, questions or bugs, please email us at
dfaust@tuebingen.mpg.de

Please cite the paper if you use the dataset for your research:

@inproceedings{dfaust:CVPR:2017,
    title = {Dynamic {FAUST}: {R}egistering Human Bodies in Motion},
    author = {Bogo, Federica and Romero, Javier and Pons-Moll, Gerard and Black, Michael J.},
    booktitle = {IEEE Conf. on Computer Vision and Pattern Recognition (CVPR)},
    month = jul,
    year = {2017},
    month_numeric = {7}
  }

==========

The data to download is organized in multiple files:

-scripts.zip
contains python scripts for loading registrations and saving them to disk as obj files, and for parsing ground-truth masks.
It also contains a subjects_and_sequences.txt file specifying subject ids, and sequences per subject, provided by the dataset.

-masks.tar.xz
contains ground-truth masks specifying, for each scan vertex, if it is accurately registered or not (according to our Evaluation, see paper).
See scripts/load_scan_mask.py for sample code parsing these files.

-registrations_{m,f}.hdf5
contain the registrations for the male (m) and female (f) subjects.
See scripts/write_sequence_to_obj.py for sample code parsing these files.
See scripts/subjects_and_sequences.txt to know the gender of each subject.

-50*tar.xz
contain the scans as ply files. We split the data into 10 files (one per subject) to simplify download.