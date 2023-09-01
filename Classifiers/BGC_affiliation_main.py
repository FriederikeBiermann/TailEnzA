import argparse
from dataset_pytorch import CustomDataset, MultiDataLoaderSampler
from torch.utils.data import DataLoader, random_split
from machine_learning_pytorch_BGC_affiliation import train_bilstm_classifier, get_model_predictions_and_labels

BGC_TYPES = ["nrp", "ripp", "pk"]
ENZYMES = ["p450", "YCAO", "SAM", "Methyl"]

def main(foldername_training_sets, batch_size, num_epochs):
    num_tags = len(BGC_TYPES)

    for enzyme in ENZYMES:
        train_dataloaders = []
        test_dataloaders = []
        for bgc_type in BGC_TYPES:
            dataset = CustomDataset(foldername_training_sets, enzyme, bgc_type)
            fragment_lengths = dataset.fragment_legths
            
            # Splitting the dataset into train/test
            train_size = int(0.8 * len(dataset))
            test_size = len(dataset) - train_size
            train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
            train_dataloaders.append(train_dataloader)
            test_dataloaders.append(test_dataloader)

        sampler = MultiDataLoaderSampler(train_dataloaders)
        combined_dataloader = DataLoader(None, batch_size=batch_size, sampler=sampler)
        
        model, epoch_losses, batch_losses = train_bilstm_classifier(combined_dataloader, fragment_lengths, num_tags, num_epochs)
        
        # Saving model with a meaningful name
        model_name = f"{enzyme}_trained_model.pt"
        torch.save(model.state_dict(), model_name)
        
        y_true, y_pred, outputs, df = get_model_predictions_and_labels(model, combined_dataloader)
        #TODO: Do some plotting

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train model for BGC affiliations')
    parser.add_argument('foldername_training_sets', type=str, help='Path to the training sets folder')
    parser.add_argument('batch_size', type=int, help='Batch size for training')
    parser.add_argument('num_epochs', type=int, help='Number of epochs for training')
    
    args = parser.parse_args()

    main(args.foldername_training_sets, args.batch_size, args.num_epochs)