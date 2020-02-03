from _loadlib.toxo_lib import *

# Pru alkaline induction at Days 0, 3, 5, and 7 (experiments 05 + 06)
htseq_input = [input_folder_dir + '180705_0601+0604+0608+0610_ns15_exon_htseq.tab.gz', input_folder_dir + '180521_0502+0503+0505+0506_ns11_exon_htseq.tab.gz']
star_input = [input_folder_dir + '180705_0601+0604+0608+0610_ns15_exon_star.tab.gz', input_folder_dir + '180521_0502+0503+0505+0506_ns11_exon_star.tab.gz']

output_dir = os.path.join(script_path, '../../Submission_analysis/Figures')+'/'
scv.settings.figdir = output_dir
scv.settings._frameon = True
sc.settings.figdir = output_dir
cell_cycle_cmap = ListedColormap(sns.color_palette(n_colors=5));
flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
louvain_cmap = ListedColormap(sns.color_palette(flatui))
