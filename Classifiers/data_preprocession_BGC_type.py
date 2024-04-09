
import os
import Bio
import pandas as pd
from Bio import SeqIO
from Bio import pairwise2
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import AlignIO
from feature_generation import *
DEBUGGING = True
# if fastas aligned before True-> use alignment made for instance in geneious utilizing MUSCLE align -> best for larger datasets
fastas_aligned_before=True
# if include_charge_features=True-> features describing general electrostatic information will be included
include_charge_features=True
#fill in filenames here!
foldernameoutput="/home/friederike/Dokumente/Arbeit/Work Friederike/Frankfurt/TailEnzA/Classifiers/Classifiers_transformer_11_03_2024_test/"
foldername_training_sets="/home/friederike/Dokumente/Arbeit/Work Friederike/Frankfurt/TailEnzA/Classifiers/Classifiers_transformer_11_03_2024_test/"


enzymes=["Methyltransf_2", "Methyltransf_3", "Methyltransf_25", "P450", "radical_SAM", "ycao", "TP_methylase"]
# For debugging
if DEBUGGING :
    enzymes=["ycao"]
BGC_types=[ "PKS", "Terpene", "Alkaloide", "NRPSs", "RiPPs"]

# the model proteins are the proteins all proteins are aligned against (beforehand) . They must be within the training data named "reference"
model_proteins_for_alignment={"P450":"PPPSLEDAAPSVLRLSPLLRELQMRAPVTKIRTPAGDEGWLVTRHAELKQLLHDERLARAHADPANAPRYVKSPLMDLLIMDDVEAARAAHAELRTLLTPQFSARRVLNMMPMVEGIAEQILNGFAAQEQPADLRGNFSLPYSLTVLCALIGIPLQEQGQLLAVLGEMATLNDAESVARSQAKLFGLLTDLAGRKRAEPGDDVISRLCETVPEDERIGPIAASLLFAGLDSVATHVDLGVVLFTQYPDQLKEALADEKLMRSGVEEILRAAKAGGSGAALPRYATDDIEIADVTIRTGDLVLLDFTLVNFDEAVFDDADLFDIRRSPNEHLTFGHGMWHCIGAPLARMMLKTAYTQLFTRLPGLKLASSVEELQVTSGQLNGGLTELPVTW",
                              "ycao": "MDIKYKLASYRICSPEETFEKIQEALKKIETVEIKNIQHLDKVNIPVYYLKRRVVVDGKEGIAIHYGKGANDIQAKVSACMEAIERFSASYDKNKVKEKPDNPINVEDLILPQYADKNVKEWVEGIDIINNETIDVPADAVFYPTSGKLFRGNTNGLASGNNLDEAILHATLEIIERDAWSLADLARKIPTKINPEDAKNPLIHELIEKYEKAGVKIILKDLTSEFEIPVVAAISDDLSKNPLMLCVGVGCHLHPEIAILRALTEVAQSRASQLHGFRRDAKLREEFTSKIPYERLKRIHRKWFEFEGEINIADMPNNARYDLKKDLKFIKDKLSEFGFDKLIYVDLNKVGVDAVRVIIPKMEVYTIDRDRLSRRAFERVKKLYY",
                              "radical_SAM": "MGSSHHHHHHSSGLVPRGSHMRTISEDILFRLEKFGGILINKTNFERIELDETEAFFLYLVQNHGIEIATSFFKKEIEMGKLERALSLNIYSDNNIEDSLNNPYETLQNARKHVAKLKKHNILSFPLELVIYPSMYCDLKCGFCFLANREDRNAKPAKDWERILRQAKDNGVLSVSILGGEPTRYFDIDNLLIACEELKIKTTITTNAQLIKKSTVEILAKSKYITPVLSLQTLDSKLNFELMGVRPDRQIKLAKYFNEVGKKCRINAVYTKQSYEQIIELVDFCIENKIDRFSVANYSEVTGYTKIKKKYDLADLRRLNEYVTDYITQREANLNFATEGCHLFTAYPELINNSIEFSEFDEMYYGCRAKYTKMEIMSNGDILPCIAFLGVNQTKQNAFEKDLLDVWYDDPLYGGIRSFRTKNSKCLSCGLLKICEGGCYVNLIKEKSPEYFRDSVCQL",
                              "Methyltransf_2":"MGSSHHHHHHSSGLVPRGSHMTVEQTPENPGTAARAAAEETVNDILQGAWKARAIHVAVELGVPELLQEGPRTATALAEATGAHEQTLRRLLRLLATVGVFDDLGHDDLFAQNALSAVLLPDPASPVATDARFQAAPWHWRAWEQLTHSVRTGEASFDVANGTSFWQLTHEDPKARELFNRAMGSVSLTEAGQVAAAYDFSGAATAVDIGGGRGSLMAAVLDAFPGLRGTLLERPPVAEEARELLTGRGLADRCEILPGDFFETIPDGADVYLIKHVLHDWDDDDVVRILRRIATAMKPDSRLLVIDNLIDERPAASTLFVDLLLLVLVGGAERSESEFAALLEKSGLRVERSLPCGAGPVRIVEIRRA",
                              "Methyltransf_3": "MSESQQLWDDVDDYFTTLLAPEDEALTAALRDSDAAGLPHINVAPNQGKLLQLLAEIQGARRILEIGTLGGYSTIWLGRALPRDGRLISFEYDAKHAEVARRNLARAGLDGISEVRVGPALESLPKLADERPEPFDLVFIDADKVNNPHYVEWALKLTRPGSLIVVDNVVRGGGVTDAGSTDPSVRGTRSALELIAEHPKLSGTAVQTVGSKGYDGFALARVLPLEHHHHHH" ,
                              "Methyltransf_25":"MAHSSATAGPQADYSGEIAELYDLVHQGKGKDYHREAADLAALVRRHSPKAASLLDVACGTGMHLRHLADSFGTVEGLELSADMLAIARRRNPDAVLHHGDMRDFSLGRRFSAVTCMFSSIGHLAGQAELDAALERFAAHVLPDGVVVVEPWWFPENFTPGYVAAGTVEAGGTTVTRVSHSSREGEATRIEVHYLVAGPDRGITHHEESHRITLFTREQYERAFTAAGLSVEFMPGGPSGRGLFTGLPGAKGETRLEHHHHHH",
                              "TP_methylase":"AMADIGSMNTTVIPPSLLDVDFPAGSVALVGAGPGDPGLLTLRAWALLQQAEVVVYDRLVARELIALLPESCQRIYVGKRCGHHSLPQEEINELLVRLARQQRRVVRLKGGDPFIFGRGAEELERLLEAGVDCQVVPGVTAASGCSTYAGIPLTHRDLAQSCTFVTGHLQNDGRLDLDWAGLARGKQTLVFYMGLGNLAEIAARLVEHGLASDTPAALVSQGTQAGQQVTRGALAELPALARRYQLKPPTLIVVGQVVALFAERAMAHPSYLGAGSPVSREAVACALEHHHHHH"}
start=0
end=350
# the splitting list defines the functional fragments that the enzymes will be cut into
splitting_lists={"P450":[["begin",1,69],["sbr1",70,78],["str1",79,195],["sbr2",196,228],["cat",229,235],["str2",236,377],["sbr3",378,383],["end",384,391]],
                 "ycao": [["begin",start,64],["sbr1",65,82],["f2",83,153],["sbr2",154,185],["f3",186,227],["sbr3",228,281],["f4",282,296],["sbr4",297,306],["f5",307,362],["sbr5",363,368],["end",369,end]],
                 "radical_SAM": [["begin",start,106],["SAM",107,310],["bridging",311,346],["end",347,end]],
                 "Methyltransf_2":[["begin",1,132],["sbr1",133,186],["SAMb",187,275],["sbr2",276,314],["sbr3",315,361],["end",362,369]],
                 "Methyltransf_3":[["begin",1,36],["sbr1",37,43],["str1",44,139],["sbr2",140,144],["str2",145,166],["sbr3",167,171],["str3",172,208],["sbr4",209,215],["end",216,224]],
                 "Methyltransf_25":[["begin",1,13],["sbr1",14,31],["SAMb",32,103],["str1",104,115],["sbr2",116,123],["str2",124,148],["sbr3",149,186],["str3",187,233],["sbr4",234,242],["end",243,250]],
                 "TP_methylase":[["begin",1,24],["sbr1",25,30],["sbr2",31,49],["sbr3",50,55],["sbr4",56,101],["sbr5",102,112],["sbr6",113,127],["sbr7",128,167],["sbr8",168,184],["sbr9",185,189],["sbr10",190,237],["sbr11",238,242],["end",243,266]]}

fragments={"P450":["begin","sbr1","str1","sbr2","cat","str2","sbr3","end"],
           "ycao":["begin","sbr1","f2","sbr2","f3","sbr3","f4","sbr4","f5","sbr5","end"],
           "radical_SAM":["begin","SAM","bridging","end"],
           "Methyltransf_2":["begin","sbr1","SAMb","sbr2","sbr3","end"],
           "Methyltransf_3":["begin","sbr1","str1","sbr2","str2","sbr3","str3","sbr4","end"],
           "Methyltransf_25":["begin","sbr1","SAMb","str1","sbr2","str2","sbr3","str3","sbr4","end"],
           "TP_methylase":["begin", "sbr1", "sbr2", "sbr3", "sbr4", "sbr5","sbr6","sbr7","sbr8", "sbr9", "sbr10", "sbr11", "end"]}

def create_filenames(enzyme, BGC_types, foldername_training_sets):
    """
    Create a dictionary of filenames categorized by BGC types for a specific enzyme.

    Parameters:
    - enzyme (str): The name of the enzyme to filter files by.
    - BGC_types (list of str): A list of biosynthetic gene cluster (BGC) types to categorize the files.
    - foldername_training_sets (str): The path to the directory containing the training set files.

    Returns:
    - dict: A dictionary where keys are BGC types, and values are lists of filenames that belong to each BGC type.
    """
    all_files = os.listdir(foldername_training_sets)
    filenames_dict = {BGC_type: [] for BGC_type in BGC_types}

    for file in all_files:
        for BGC_type in BGC_types:
            if f"{enzyme}_" in file and BGC_type in file:
                filenames_dict[BGC_type].append(os.path.join(foldername_training_sets, file))
    # For debugging
    if DEBUGGING :
        filenames_dict = {"RiPPs": ["toy_ripp_YCAO alignment.fas"]}
    
    return filenames_dict



def process_datasets(foldername_training_sets, model, batch_converter, include_charge_features=True):
    """
    Process datasets to generate a feature matrix for each enzyme-BGC type pair.

    Parameters:
    - foldername_training_sets (str): The path to the directory containing the training set files.
    - model: The pre-trained transformer model used for generating sequence embeddings.
    - batch_converter: A utility function provided by the transformer model for converting sequences into a compatible format.
    - include_charge_features (bool): Flag to indicate whether to include electrostatic charge features in the feature matrix.

    Returns:
    - None: The function saves the complete feature matrix to a CSV file and prints the path to this file.
    """
    complete_feature_matrix = pd.DataFrame()
    
    for enzyme in enzymes:
        filenames_dict = create_filenames(enzyme, BGC_types, foldername_training_sets)
        
        for BGC_type, datasets in filenames_dict.items():
            for dataset in datasets:
                msa_path = Path(dataset)
                splitting_list = splitting_lists[enzyme]
                fragment_matrix = fragment_alignment(msa_path, splitting_list, fastas_aligned_before)
                feature_matrix = featurize_fragments(fragment_matrix, batch_converter, model, include_charge_features)
                feature_matrix["target"] = BGC_type
                complete_feature_matrix = pd.concat([complete_feature_matrix, feature_matrix], ignore_index=True)
    
    output_path = Path(foldername_output, "complete_feature_matrix.csv")
    complete_feature_matrix.to_csv(output_path, index=False)
    print(f"Feature matrix saved to {output_path}")
