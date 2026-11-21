import torch


@torch.no_grad()
def _anchor_assignments(model, control):
    means, _ = model.encode(control)
    distances = torch.cdist(means, model.anchors)
    return distances.argmin(dim=1), distances.argmin(dim=0)


@torch.no_grad()
def _refresh_reference(model, control, reference, groups, nearest, momentum):
    control_means, _ = model.encode(control)
    reference_means, _ = model.encode(reference)

    centers = control_means[nearest].clone()
    for k in range(model.n_anchors):
        members = groups == k
        if members.any():
            centers[k] = control_means[members].mean(dim=0)

    assignments = torch.cdist(reference_means, centers).argmin(dim=1)
    for k in range(model.n_anchors):
        members = assignments == k
        if members.any():
            centers[k].lerp_(reference_means[members].mean(dim=0), momentum)

    model.anchors.copy_(centers)
    signals = model._signals(reference)
    model.risk_low.copy_(torch.quantile(signals, 0.05, dim=0))
    model.risk_high.copy_(torch.quantile(signals, 0.95, dim=0))


def selective_adaptation(model, control, control_mask, confirmed_normal, *,
                         keep_threshold=0.5, min_keep=2, steps=3,
                         learning_rate=1e-3, lambda_reg=1e-2, momentum=0.05):
    """Update ICA using retained-sample importance and confirmed normal data."""
    if not model.reference_ready.item():
        raise RuntimeError("Call fit_reference() before adaptation.")
    if control_mask is None:
        return control, False
    if control.ndim != 2 or confirmed_normal.ndim != 2:
        raise ValueError("Expected two-dimensional feature matrices.")
    if control.shape[1] != confirmed_normal.shape[1]:
        raise ValueError("Control and new samples must have matching features.")
    if control_mask.shape != (len(control),):
        raise ValueError("Expected one retention weight per control sample.")
    if not torch.isfinite(control_mask).all() or (
        (control_mask < 0) | (control_mask > 1)
    ).any():
        raise ValueError("Retention weights must be finite and in [0, 1].")
    if not 0 <= keep_threshold <= 1 or not 0 <= momentum <= 1:
        raise ValueError("Threshold and momentum must be in [0, 1].")
    if min_keep < 1 or steps < 1 or learning_rate <= 0 or lambda_reg < 0:
        raise ValueError("Invalid adaptation settings.")

    keep = control_mask >= keep_threshold
    retained = control[keep]
    if len(retained) < min_keep or len(confirmed_normal) == 0:
        return control, False

    with torch.no_grad():
        _, confidence = model(retained)
        weights = control_mask[keep].detach() * confidence
        groups, nearest = _anchor_assignments(model, control)

    parameters = [p for p in model.parameters() if p.requires_grad]
    previous = [p.detach().clone() for p in parameters]
    importance = [torch.zeros_like(p) for p in parameters]

    with torch.enable_grad():
        for sample, weight in zip(retained, weights):
            loss = model.training_loss(sample.unsqueeze(0))
            gradients = torch.autograd.grad(loss, parameters)
            for omega, gradient in zip(importance, gradients):
                omega.add_(weight * gradient.detach().square())

        for omega in importance:
            omega.div_(len(retained))

        optimizer = torch.optim.Adam(parameters, lr=learning_rate)
        for _ in range(steps):
            optimizer.zero_grad()
            new_loss = model.training_loss(confirmed_normal) * len(confirmed_normal)
            penalty = sum(
                (omega * (parameter - old).square()).sum()
                for parameter, old, omega in zip(parameters, previous, importance)
            )
            (new_loss + lambda_reg * penalty).backward()
            optimizer.step()

    reference = torch.cat((retained, confirmed_normal), dim=0).detach()
    _refresh_reference(model, control, reference, groups, nearest, momentum)
    return reference, True