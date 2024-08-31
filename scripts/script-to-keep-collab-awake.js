function keepSessionActive() {
    console.log(`Interval ID: ${intervalId} - Attempting to click the 'Files' button using parent element and shadow DOM.`);
    const parentElement = document.querySelector("body > div.notebook-vertical > div.notebook-horizontal > colab-left-pane > div > div.left-pane-top > div:nth-child(5) > md-icon-button");

    if (parentElement && parentElement.shadowRoot) {
        const folderButton = parentElement.shadowRoot.querySelector("#button > span.touch");

        if (folderButton) {
            folderButton.click();
            console.log(`Interval ID: ${intervalId} - 'Files' button clicked to open.`);

            setTimeout(() => {
                folderButton.click();
                console.log(`Interval ID: ${intervalId} - 'Files' button clicked to close.`);
            }, 1000);
        } else {
            console.log(`Interval ID: ${intervalId} - Folder button not found.`);
        }
    } else {
        console.log(`Interval ID: ${intervalId} - Parent element or shadow DOM not found.`);
    }
}

const intervalId = setInterval(keepSessionActive, 30000);
// clearInterval(intervalId);
