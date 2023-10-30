
import os
import Bio
import pandas as pd
from Bio import SeqIO
from Bio import pairwise2
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import AlignIO
from feature_generation import *
# if fastas aligned before True-> use alignment made for instance in geneious utilizing MUSCLE align -> best for larger datasets
fastas_aligned_before=True
# if include_charge_features=True-> features describing general electrostatic information will be included
include_charge_features=False
#fill in filenames here!
foldernameoutput="/beegfs/projects/p450/Output_training_dataset_neu_25/data_preprocession_BGC_type/output/"
foldername_training_sets="/beegfs/projects/p450/Output_training_dataset_neu_25/data_preprocession_BGC_type/input_align_align/"
filename_permutations="permutations.txt"

#enzymes=["p450","ycao","SAM","Methyl"]
enzymes=["Methyltransf_2", "Methyltransf_3", "Methyltransf_25", "P450", "radical_SAM", "ycao"]
#enzymes=["ycao"]
BGC_types=[ "PKS", "Terpene", "Alkaloide", "NRPSs", "RiPPs"]
#BGC_types=["ripp"]
# the model proteins are the proteins all proteins are aligned against (beforehand) . They must be within the training data named "reference"
model_proteins_for_alignment={"P450":"PPPSLEDAAPSVLRLSPLLRELQMRAPVTKIRTPAGDEGWLVTRHAELKQLLHDERLARAHADPANAPRYVKSPLMDLLIMDDVEAARAAHAELRTLLTPQFSARRVLNMMPMVEGIAEQILNGFAAQEQPADLRGNFSLPYSLTVLCALIGIPLQEQGQLLAVLGEMATLNDAESVARSQAKLFGLLTDLAGRKRAEPGDDVISRLCETVPEDERIGPIAASLLFAGLDSVATHVDLGVVLFTQYPDQLKEALADEKLMRSGVEEILRAAKAGGSGAALPRYATDDIEIADVTIRTGDLVLLDFTLVNFDEAVFDDADLFDIRRSPNEHLTFGHGMWHCIGAPLARMMLKTAYTQLFTRLPGLKLASSVEELQVTSGQLNGGLTELPVTW",
                              "ycao": "MDIKYKLASYRICSPEETFEKIQEALKKIETVEIKNIQHLDKVNIPVYYLKRRVVVDGKEGIAIHYGKGANDIQAKVSACMEAIERFSASYDKNKVKEKPDNPINVEDLILPQYADKNVKEWVEGIDIINNETIDVPADAVFYPTSGKLFRGNTNGLASGNNLDEAILHATLEIIERDAWSLADLARKIPTKINPEDAKNPLIHELIEKYEKAGVKIILKDLTSEFEIPVVAAISDDLSKNPLMLCVGVGCHLHPEIAILRALTEVAQSRASQLHGFRRDAKLREEFTSKIPYERLKRIHRKWFEFEGEINIADMPNNARYDLKKDLKFIKDKLSEFGFDKLIYVDLNKVGVDAVRVIIPKMEVYTIDRDRLSRRAFERVKKLYY",
                              "radical_SAM": "MGSSHHHHHHSSGLVPRGSHMRTISEDILFRLEKFGGILINKTNFERIELDETEAFFLYLVQNHGIEIATSFFKKEIEMGKLERALSLNIYSDNNIEDSLNNPYETLQNARKHVAKLKKHNILSFPLELVIYPSMYCDLKCGFCFLANREDRNAKPAKDWERILRQAKDNGVLSVSILGGEPTRYFDIDNLLIACEELKIKTTITTNAQLIKKSTVEILAKSKYITPVLSLQTLDSKLNFELMGVRPDRQIKLAKYFNEVGKKCRINAVYTKQSYEQIIELVDFCIENKIDRFSVANYSEVTGYTKIKKKYDLADLRRLNEYVTDYITQREANLNFATEGCHLFTAYPELINNSIEFSEFDEMYYGCRAKYTKMEIMSNGDILPCIAFLGVNQTKQNAFEKDLLDVWYDDPLYGGIRSFRTKNSKCLSCGLLKICEGGCYVNLIKEKSPEYFRDSVCQL",
                              "Methyltransf_2":"MGSSHHHHHHSSGLVPRGSHMTVEQTPENPGTAARAAAEETVNDILQGAWKARAIHVAVELGVPELLQEGPRTATALAEATGAHEQTLRRLLRLLATVGVFDDLGHDDLFAQNALSAVLLPDPASPVATDARFQAAPWHWRAWEQLTHSVRTGEASFDVANGTSFWQLTHEDPKARELFNRAMGSVSLTEAGQVAAAYDFSGAATAVDIGGGRGSLMAAVLDAFPGLRGTLLERPPVAEEARELLTGRGLADRCEILPGDFFETIPDGADVYLIKHVLHDWDDDDVVRILRRIATAMKPDSRLLVIDNLIDERPAASTLFVDLLLLVLVGGAERSESEFAALLEKSGLRVERSLPCGAGPVRIVEIRRA",
                              "Methyltransf_3": "MSESQQLWDDVDDYFTTLLAPEDEALTAALRDSDAAGLPHINVAPNQGKLLQLLAEIQGARRILEIGTLGGYSTIWLGRALPRDGRLISFEYDAKHAEVARRNLARAGLDGISEVRVGPALESLPKLADERPEPFDLVFIDADKVNNPHYVEWALKLTRPGSLIVVDNVVRGGGVTDAGSTDPSVRGTRSALELIAEHPKLSGTAVQTVGSKGYDGFALARVLPLEHHHHHH" ,
                              "Methyltransf_25":"MAHSSATAGPQADYSGEIAELYDLVHQGKGKDYHREAADLAALVRRHSPKAASLLDVACGTGMHLRHLADSFGTVEGLELSADMLAIARRRNPDAVLHHGDMRDFSLGRRFSAVTCMFSSIGHLAGQAELDAALERFAAHVLPDGVVVVEPWWFPENFTPGYVAAGTVEAGGTTVTRVSHSSREGEATRIEVHYLVAGPDRGITHHEESHRITLFTREQYERAFTAAGLSVEFMPGGPSGRGLFTGLPGAKGETRLEHHHHHH" }
start=0
end=350
# the splitting list defines the functional fragments that the enzymes will be cut into
splitting_lists={"P450":[["begin",1,69],["sbr1",70,78],["str1",79,195],["sbr2",196,228],["cat",229,235],["str2",236,377],["sbr3",378,383],["end",384,391]],
                 "ycao": [["begin",start,64],["sbr1",65,82],["f2",83,153],["sbr2",154,185],["f3",186,227],["sbr3",228,281],["f4",282,296],["sbr4",297,306],["f5",307,362],["sbr5",363,368],["end",369,end]],
                 "radical_SAM": [["begin",start,106],["SAM",107,310],["bridging",311,346],["end",347,end]],
                 "Methyltransf_2":[["begin",1,132],["sbr1",133,186],["SAMb",187,275],["sbr2",276,314],["sbr3",315,361],["end",362,369]],
                 "Methyltransf_3":[["begin",1,36],["sbr1",37,43],["str1",44,139],["sbr2",140,144],["str2",145,166],["sbr3",167,171],["str3",172,208],["sbr4",209,215],["end",216,224]],
                 "Methyltransf_25":[["begin",1,13],["sbr1",14,31],["SAMb",32,103],["str1",104,115],["sbr2",116,123],["str2",124,148],["sbr3",149,186],["str3",187,233],["sbr4",234,242],["end",243,250]]}

fragments={"P450":["begin","sbr1","str1","sbr2","cat","str2","sbr3","end"],
           "ycao":["begin","sbr1","f2","sbr2","f3","sbr3","f4","sbr4","f5","sbr5","end"],
           "radical_SAM":["begin","SAM","bridging","end"],
           "Methyltransf_2":["begin","sbr1","SAMb","sbr2","sbr3","end"],
           "Methyltransf_3":["begin","sbr1","str1","sbr2","str2","sbr3","str3","sbr4","end"],
           "Methyltransf_25":["begin","sbr1","SAMb","str1","sbr2","str2","sbr3","str3","sbr4","end"]}


#permutations= list of 4-aa motifs to use as features
with open(filename_permutations, 'r') as file:
    permutations = [line.rstrip('\n') for line in file]


def create_filenames(enzyme, BGC_types, foldername_training_sets):
    # List all files in the directory
    all_files = os.listdir(foldername_training_sets)

    # Initialize a dictionary to store filenames for each BGC type
    filenames_dict = {BGC_type: [] for BGC_type in BGC_types}

    # Populate the dictionary based on the enzyme and BGC types
    for file in all_files:
        for BGC_type in BGC_types:
            if f"{enzyme}_" in file and BGC_type in file:
                filenames_dict[BGC_type].append(
                    os.path.join(foldername_training_sets, file))

    return filenames_dict




for enzyme in enzymes:
    filenames_dict = create_filenames(
        enzyme, BGC_types, foldername_training_sets)
    complete_feature_matrix = pd.DataFrame()
    path_complete_feature_matrix = foldernameoutput + \
        enzyme + "_complete_feature_matrix.csv"
    for BGC_type, datasets in filenames_dict.items():
        for dataset in datasets:
            # Fill in the feature matrix with data from files. The fragment matrix is a table with all sequences split into the different functional parts.
            if fastas_aligned_before:
                alignment = AlignIO.read(open(dataset), "fasta")
                print(dataset)
                fragment_matrix = fragment_alignment(
                    alignment, splitting_lists[enzyme], fastas_aligned_before)
            else:
                fragment_matrix = pd.DataFrame()
                seq_record_ids = []
                for seq_record in SeqIO.parse(dataset, "fasta"):
                    alignment = pairwise2.align.globalmc(
                        model_proteins_for_alignment[enzyme], seq_record.seq, 1, -1, -8, -2)
                    fragment_matrix_for_record = fragment_alignment(
                        alignment[0], splitting_lists[enzyme], fastas_aligned_before)
                    fragment_matrix = fragment_matrix.append(
                        fragment_matrix_for_record, ignore_index=True)
                    seq_record_ids.append(seq_record.id)

            # Obtain features from each fragment
            feature_matrix = featurize(
                fragment_matrix, permutations, fragments[enzyme], include_charge_features)

            # Set the true type of BGC for each BGC as target
            feature_matrix["target"] = BGC_type
            complete_feature_matrix = complete_feature_matrix.append(
                feature_matrix, ignore_index=True)
    print(complete_feature_matrix)
    complete_feature_matrix.to_csv(path_complete_feature_matrix, index=False)
