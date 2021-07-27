def qmargin_accumulate(loader: DataLoader, model: nn.Module, batch_size: int, augmentation: nn.Module,
                       q: float = 1.0, margin: float = 0.4) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        accumulated_data: List[torch.Tensor] = []
        accumulated_labels: List[torch.Tensor] = []
        current_accumulation = 0
        samples_left = len(loader.dataset)

        model.eval()
        for batch in loader:
            x, y = augment_batch(batch, augmentation)
            with torch.no_grad():
                model_output = model(x)
                beliefs = torch.softmax(model_output, dim=1)
                del model_output
                sorted_beliefs = torch.sort(beliefs, dim=1).values

                # the highest belief per sample -> i.e. which class samples would be classified as
                top = sorted_beliefs[:, -1] # top class belief strengths

                num_classes = len(beliefs[0]) - 1

                # get the belief index from q, between [0..n-1] of the n belief-sorted classes
                q_class = int(q * (num_classes - 1) + 0.5)  # q_class \in n, n != top

                qs = sorted_beliefs[:, q_class] # q class belief strengths
                decision_distance = top - qs    # how far apart the belief strengths are (decision margin/boundary)
                # mask of samples where the decision margin is too narrow
                inside_margin = torch.nonzero(decision_distance < margin).view(-1)  
                del beliefs, top, qs, sorted_beliefs, decision_distance

		# Only keep samples where the decision margin is too narrow, other samples are already well separated.
                accumulated_data.append(x[inside_margin])
                accumulated_labels.append(y[inside_margin])
                current_accumulation += len(inside_margin)
                del inside_margin

            if current_accumulation >= batch_size:
                model.train()
                yield torch.cat(accumulated_data)[:batch_size], torch.cat(accumulated_labels)[:batch_size]
                model.eval()
                samples_left -= current_accumulation

                current_accumulation = 0
                accumulated_data.clear()
                accumulated_labels.clear()

                if samples_left < batch_size:
                    model.train()
                    return
        model.train()