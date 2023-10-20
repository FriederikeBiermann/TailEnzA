enzymes = {
    "p450": {
        "splitting_list": {
            "begin": [0, 92],
            "sbr1": [93, 192],
            "sbr2": [193, 275],
            "core": [276, 395],
            "end": [396, 350],
            "fes1": [54, 115],
            "fes2": [302, 401]
        },
        "hmm_file": "p450.hmm",
        "classifier_enzyme": "_AdaBoostClassifier_classifier.sav",
        "classifier_metabolism": "_ExtraTreesClassifier_classifier.sav",
        "reference_for_alignment": "MSAVALPRVSGGHDEHGHLEEFRTDPIGLMQRVRDECGDVGTFQLAGKQVVLLSGSHANEFFFRAGDDDLDQAKAYPFMTPIFGEGVVFDASPERRKEMLHNAALRGEQMKGHAATIEDQVRRMIADWGEAGEIDLLDFFAELTIYTSSACLIGKKFRDQLDGRFAKLYHELERGTDPLAYVDPYLPIESLRRRDEARNGLVALVADIMNGRIANPPTDKSDRDMLDVLIAVKAETGTPRFSADEITGMFISMMFAGHHTSSGTASWTLIELMRHRDAYAAVIDELDELYGDGRSVSFHLRQIPQLENVLKETLRLHPPLIILMRVAKGEFEVQGHRIHEGDLVAASPAISNRIPEDFPDPHDFVPARYEQPRQEDLLNRWTWIPFGAGRHRCVGAAFAIMQIKAIFSVLLREYEFEMAQPPESYRNDHSKMVVQLAQPACVRYRRRTGV",
        "gap_opening_penalty": -1,
        "gap_extend_penalty": -1
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
            "end": [369, 350]
        },
        "hmm_file": "ycao.hmm",
        "classifier_enzyme": "_ExtraTreesClassifier_classifier.sav",
        "classifier_metabolism": "_AdaBoostClassifier_classifier.sav",
        "reference_for_alignment": "MDIKYKLASYRICSPEETFEKIQEALKKIETVEIKNIQHLDKVNIPVYYLKRRVVVDGKEGIAIHYGKGANDIQAKVSACMEAIERFSASYDKNKVKEKPDNPINVEDLILPQYADKNVKEWVEGIDIINNETIDVPADAVFYPTSGKLFRGNTNGLASGNNLDEAILHATLEIIERDAWSLADLARKIPTKINPEDAKNPLIHELIEKYEKAGVKIILKDLTSEFEIPVVAAISDDLSKNPLMLCVGVGCHLHPEIAILRALTEVAQSRASQLHGFRRDAKLREEFTSKIPYERLKRIHRKWFEFEGEINIADMPNNARYDLKKDLKFIKDKLSEFGFDKLIYVDLNKVGVDAVRVIIPKMEVYTIDRDRLSRRAFERVKKLYY",
        "gap_opening_penalty": -1,
        "gap_extend_penalty": -1
    },
    "SAM": {
        "splitting_list": {
            "begin": [0, 106],
            "SAM": [107, 310],
            "bridging": [311, 346],
            "end": [347, 350]
        },
        "hmm_file": "SAM.hmm",
        "classifier_enzyme": "_ExtraTreesClassifier_classifier.sav",
        "classifier_metabolism": "_AdaBoostClassifier_classifier.sav",
        "reference_for_alignment": "MGSSHHHHHHSSGLVPRGSHMRTISEDILFRLEKFGGILINKTNFERIELDETEAFFLYLVQNHGIEIATSFFKKEIEMGKLERALSLNIYSDNNIEDSLNNPYETLQNARKHVAKLKKHNILSFPLELVIYPSMYCDLKCGFCFLANREDRNAKPAKDWERILRQAKDNGVLSVSILGGEPTRYFDIDNLLIACEELKIKTTITTNAQLIKKST",
        "gap_opening_penalty": -1,
        "gap_extend_penalty": -1
    }
}
 
