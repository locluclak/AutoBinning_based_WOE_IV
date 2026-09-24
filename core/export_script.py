EXPORT_SCRIPT = """function exportConfig() {
    const selectedConfig = {};
    const radios = document.querySelectorAll('input[type="radio"]:checked');
    radios.forEach(radio => {
        const featureName = radio.name;
        const selectedOption = radio.getAttribute('data-option');
        const part = radio.getAttribute('data-part');
        const ftype = radio.getAttribute('data-type');
        const splitsData = JSON.parse(radio.getAttribute('data-splits'));
        let splits = splitsData;
        let note = null;
        const roundingRow = radio.closest('td').querySelector('.option-splits');
        if (roundingRow && roundingRow.getAttribute('data-rounded') === 'true') {
            const direction = roundingRow.querySelector('.rounding-direction').value;
            const precision = parseFloat(roundingRow.querySelector('.rounding-precision').value);
            const rounded = splitsData.map(v => roundSplit(Number(v), direction, precision));
            const modified = rounded.some((v, i) => v !== Number(splitsData[i]));
            if (modified) {
                splits = rounded;
                note = 'Splits have been modified by rounding';
            }
        }
        const feature = {
            option: selectedOption,
            part: part,
            type: ftype,
            splits: splits
        };
        if (note) feature.note = note;
        selectedConfig[featureName] = feature;
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

function exportSelectedHTML() {
    const clone = document.documentElement.cloneNode(true);
    clone.querySelectorAll('section.feature').forEach(section => {
        const radio = section.querySelector('input[type="radio"]:checked');
        if (!radio) {
            section.remove();
            return;
        }
        const selectedPart = radio.getAttribute('data-part');
        const selectedOption = radio.getAttribute('data-option');
        const statusTable = section.querySelector('table.status-table');
        if (statusTable) statusTable.remove();
        section.querySelectorAll('.plot-part').forEach(pp => {
            if (pp.getAttribute('data-part') !== selectedPart) {
                pp.remove();
                return;
            }
            pp.querySelectorAll('.plot-figure').forEach(fig => {
                if (fig.getAttribute('data-option') !== selectedOption) fig.remove();
            });
        });
        section.querySelectorAll('.woe-row').forEach(row => {
            const part = row.getAttribute('data-part');
            if (part === null) return;
            if (part !== selectedPart) {
                row.remove();
                return;
            }
            row.querySelectorAll('.woe-block').forEach(block => {
                if (block.getAttribute('data-option') !== selectedOption) block.remove();
            });
        });
        section.style.display = '';
    });
    const html = '<!DOCTYPE html>\\n' + clone.outerHTML;
    const blob = new Blob([html], {type: 'text/html;charset=utf-8'});
    const downloadAnchor = document.createElement('a');
    downloadAnchor.setAttribute("href", URL.createObjectURL(blob));
    downloadAnchor.setAttribute("download", "selected_report.html");
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
    URL.revokeObjectURL(downloadAnchor.href);
}
"""