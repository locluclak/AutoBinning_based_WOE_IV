EXPORT_SCRIPT = """function exportConfig() {
    const selectedConfig = {};
    const radios = document.querySelectorAll('input[type="radio"]:checked');
    radios.forEach(radio => {
        const featureName = radio.name;
        const selectedOption = radio.getAttribute('data-option');
        const splitsData = JSON.parse(radio.getAttribute('data-splits'));
        selectedConfig[featureName] = {
            option: selectedOption,
            splits: splitsData
        };
    });
    const output = {
        config: APP_CONFIG,
        features: selectedConfig
    };
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(output, null, 4));
    const downloadAnchor = document.createElement('a');
    downloadAnchor.setAttribute("href", dataStr);
    downloadAnchor.setAttribute("download", "selected_feature_splits.json");
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
}
"""