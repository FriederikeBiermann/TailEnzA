BGC_types = ["PKS", "Terpene", "Alkaloid", "NRPS", "RiPP"]
enzymes = {
    "P450": {
        "splitting_list": {
            "begin": [1, 69],
            "sbr1": [70, 78],
            "str1": [79, 195],
            "sbr2": [196, 228],
            "cat": [229, 235],
            "str2": [236, 377],
            "sbr3": [378, 383],
            "end": [384, 391],
        },
        "hmm_file": "p450.hmm",
        "classifier_BGC_type": "_AdvancedFFNN/AdvancedFFNN_model.pth",
        "classifier_metabolism": "_LSTM/LSTM_model.pth",
        "reference_for_alignment": "PPPSLEDAAPSVLRLSPLLRELQMRAPVTKIRTPAGDEGWLVTRHAELKQLLHDERLARAHADPANAPRYVKSPLMDLLIMDDVEAARAAHAELRTLLTPQFSARRVLNMMPMVEGIAEQILNGFAAQEQPADLRGNFSLPYSLTVLCALIGIPLQEQGQLLAVLGEMATLNDAESVARSQAKLFGLLTDLAGRKRAEPGDDVISRLCETVPEDERIGPIAASLLFAGLDSVATHVDLGVVLFTQYPDQLKEALADEKLMRSGVEEILRAAKAGGSGAALPRYATDDIEIADVTIRTGDLVLLDFTLVNFDEAVFDDADLFDIRRSPNEHLTFGHGMWHCIGAPLARMMLKTAYTQLFTRLPGLKLASSVEELQVTSGQLNGGLTELPVTW",
        "gap_opening_penalty": -2,
        "gap_extend_penalty": -1,
        "center": -1,
        "min_length": 200,
        "max_length": 800,
    },
    "ycao": {
        "splitting_list": {
            "begin": [0, 64],
            "sbr1": [65, 82],
            "f2": [83, 153],
            "sbr2": [154, 185],
            "f3": [186, 227],
            "sbr3": [228, 281],
            "f4": [282, 296],
            "sbr4": [297, 306],
            "f5": [307, 362],
            "sbr5": [363, 368],
            "end": [369, 350],
        },
        "hmm_file": "ycao.hmm",
        "classifier_BGC_type": "",
        "classifier_metabolism": "_BasicFFNN/BasicFFNN_model.pth",
        "reference_for_alignment": "MDIKYKLASYRICSPEETFEKIQEALKKIETVEIKNIQHLDKVNIPVYYLKRRVVVDGKEGIAIHYGKGANDIQAKVSACMEAIERFSASYDKNKVKEKPDNPINVEDLILPQYADKNVKEWVEGIDIINNETIDVPADAVFYPTSGKLFRGNTNGLASGNNLDEAILHATLEIIERDAWSLADLARKIPTKINPEDAKNPLIHELIEKYEKAGVKIILKDLTSEFEIPVVAAISDDLSKNPLMLCVGVGCHLHPEIAILRALTEVAQSRASQLHGFRRDAKLREEFTSKIPYERLKRIHRKWFEFEGEINIADMPNNARYDLKKDLKFIKDKLSEFGFDKLIYVDLNKVGVDAVRVIIPKMEVYTIDRDRLSRRAFERVKKLYY",
        "gap_opening_penalty": -2,
        "gap_extend_penalty": -1,
        "center": -5,
        "min_length": 200,
        "max_length": 800,
    },
    "radical_SAM": {
        "splitting_list": {
            "begin": [0, 106],
            "SAM": [107, 310],
            "bridging": [311, 346],
            "end": [347, 350],
        },
        "hmm_file": "radical_SAM.hmm",
        "classifier_BGC_type": "_LSTM/LSTM_model.pth",
        "classifier_metabolism": "_IntermediateFFNN/IntermediateFFNN_model.pth",
        "reference_for_alignment": "MGSSHHHHHHSSGLVPRGSHMRTISEDILFRLEKFGGILINKTNFERIELDETEAFFLYLVQNHGIEIATSFFKKEIEMGKLERALSLNIYSDNNIEDSLNNPYETLQNARKHVAKLKKHNILSFPLELVIYPSMYCDLKCGFCFLANREDRNAKPAKDWERILRQAKDNGVLSVSILGGEPTRYFDIDNLLIACEELKIKTTITTNAQLIKKSTVEILAKSKYITPVLSLQTLDSKLNFELMGVRPDRQIKLAKYFNEVGKKCRINAVYTKQSYEQIIELVDFCIENKIDRFSVANYSEVTGYTKIKKKYDLADLRRLNEYVTDYITQREANLNFATEGCHLFTAYPELINNSIEFSEFDEMYYGCRAKYTKMEIMSNGDILPCIAFLGVNQTKQNAFEKDLLDVWYDDPLYGGIRSFRTKNSKCLSCGLLKICEGGCYVNLIKEKSPEYFRDSVCQL",
        "gap_opening_penalty": -2,
        "gap_extend_penalty": -1,
        "center": -2,
        "min_length": 200,
        "max_length": 600,
    },
    "Methyltransf_2": {
        "splitting_list": {
            "begin": [1, 132],
            "sbr1": [133, 186],
            "SAMb": [187, 275],
            "sbr2": [276, 314],
            "sbr3": [315, 361],
            "end": [362, 369],
        },
        "hmm_file": "Methyltransf_2.hmm",
        "classifier_BGC_type": "_LSTM/LSTM_model.pth",
        "classifier_metabolism": "_LSTM/LSTM_model.pth",
        "reference_for_alignment": "MGSSHHHHHHSSGLVPRGSHMTVEQTPENPGTAARAAAEETVNDILQGAWKARAIHVAVELGVPELLQEGPRTATALAEATGAHEQTLRRLLRLLATVGVFDDLGHDDLFAQNALSAVLLPDPASPVATDARFQAAPWHWRAWEQLTHSVRTGEASFDVANGTSFWQLTHEDPKARELFNRAMGSVSLTEAGQVAAAYDFSGAATAVDIGGGRGSLMAAVLDAFPGLRGTLLERPPVAEEARELLTGRGLADRCEILPGDFFETIPDGADVYLIKHVLHDWDDDDVVRILRRIATAMKPDSRLLVIDNLIDERPAASTLFVDLLLLVLVGGAERSESEFAALLEKSGLRVERSLPCGAGPVRIVEIRRA",
        "gap_opening_penalty": -1,
        "gap_extend_penalty": -2,
        "center": 0,
        "min_length": 200,
        "max_length": 800,
    },
    "Methyltransf_3": {
        "splitting_list": {
            "begin": [1, 36],
            "sbr1": [37, 43],
            "str1": [44, 139],
            "sbr2": [140, 144],
            "str2": [145, 166],
            "sbr3": [167, 171],
            "str3": [172, 208],
            "sbr4": [209, 215],
            "end": [216, 224],
        },
        "hmm_file": "Methyltransf_3.hmm",
        "classifier_BGC_type": "_AdvancedFFNN/AdvancedFFNN_model.pth",
        "classifier_metabolism": "_LSTM/LSTM_model.pth",
        "reference_for_alignment": "MSESQQLWDDVDDYFTTLLAPEDEALTAALRDSDAAGLPHINVAPNQGKLLQLLAEIQGARRILEIGTLGGYSTIWLGRALPRDGRLISFEYDAKHAEVARRNLARAGLDGISEVRVGPALESLPKLADERPEPFDLVFIDADKVNNPHYVEWALKLTRPGSLIVVDNVVRGGGVTDAGSTDPSVRGTRSALELIAEHPKLSGTAVQTVGSKGYDGFALARVLPLEHHHHHH",
        "gap_opening_penalty": -1,
        "gap_extend_penalty": -2,
        "center": 0,
        "min_length": 200,
        "max_length": 800,
    },
    "Methyltransf_25": {
        "splitting_list": {
            "begin": [1, 13],
            "sbr1": [14, 31],
            "SAMb": [32, 103],
            "str1": [104, 115],
            "sbr2": [116, 123],
            "str2": [124, 148],
            "sbr3": [149, 186],
            "str3": [187, 233],
            "sbr4": [234, 242],
            "end": [243, 250],
        },
        "hmm_file": "Methyltransf_25.hmm",
        "classifier_BGC_type": "_LSTM/LSTM_model.pth",
        "classifier_metabolism": "_AdvancedFFNN/AdvancedFFNN_model.pth",
        "reference_for_alignment": "MAHSSATAGPQADYSGEIAELYDLVHQGKGKDYHREAADLAALVRRHSPKAASLLDVACGTGMHLRHLADSFGTVEGLELSADMLAIARRRNPDAVLHHGDMRDFSLGRRFSAVTCMFSSIGHLAGQAELDAALERFAAHVLPDGVVVVEPWWFPENFTPGYVAAGTVEAGGTTVTRVSHSSREGEATRIEVHYLVAGPDRGITHHEESHRITLFTREQYERAFTAAGLSVEFMPGGPSGRGLFTGLPGAKGETRLEHHHHHH",
        "gap_opening_penalty": -1,
        "gap_extend_penalty": -2,
        "center": 0,
        "min_length": 200,
        "max_length": 800,
    },
    "TP_methylase": {
        "splitting_list": {
            "begin": [1, 24],
            "sbr1": [25, 30],
            "sbr2": [31, 49],
            "sbr3": [50, 55],
            "sbr4": [56, 101],
            "sbr5": [102, 112],
            "sbr6": [113, 127],
            "sbr7": [128, 167],
            "sbr8": [168, 184],
            "sbr9": [185, 189],
            "sbr10": [190, 237],
            "sbr11": [238, 242],
            "end": [243, 266],
        },
        "hmm_file": "TP_methylase.hmm",
        "classifier_BGC_type": "_LSTM/LSTM_model.pth",
        "classifier_metabolism": "_IntermediateFFNN/IntermediateFFNN_model.pth",
        "reference_for_alignment": "AMADIGSMNTTVIPPSLLDVDFPAGSVALVGAGPGDPGLLTLRAWALLQQAEVVVYDRLVARELIALLPESCQRIYVGKRCGHHSLPQEEINELLVRLARQQRRVVRLKGGDPFIFGRGAEELERLLEAGVDCQVVPGVTAASGCSTYAGIPLTHRDLAQSCTFVTGHLQNDGRLDLDWAGLARGKQTLVFYMGLGNLAEIAARLVEHGLASDTPAALVSQGTQAGQQVTRGALAELPALARRYQLKPPTLIVVGQVVALFAERAMAHPSYLGAGSPVSREAVACALEHHHHHH",
        "gap_opening_penalty": -1,
        "gap_extend_penalty": -2,
        "center": 0,
        "min_length": 200,
        "max_length": 800,
    },
}
