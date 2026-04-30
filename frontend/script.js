let historyData = JSON.parse(localStorage.getItem("analysisHistory")) || [];
let batchFiles = [];
let isBatchMode = false;

async function analyzeImage() {
    const fileInput = document.getElementById("fileInput");
    const file = fileInput.files[0];

    if (!file) {
        alert("Please upload an MRI image first.");
        return;
    }

    // Show loading state
    document.getElementById("inputImage").src = "";
    document.getElementById("maskImage").src = "";
    document.getElementById("overlayImage").src = "";
    
    // Create form data for API call
    const formData = new FormData();
    formData.append('file', file);

    try {
        // Call backend API
        const response = await fetch('http://127.0.0.1:8001/predict', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();
        
        // Display the input image and save to history
        const reader = new FileReader();
        reader.onload = function (e) {
            const imageSrc = e.target.result;
            document.getElementById("inputImage").src = imageSrc;
            
            // Save history with results
            const entry = {
                image: imageSrc,
                maskImage: `http://127.0.0.1:8001/${result.mask_image}`,
                overlayImage: `http://127.0.0.1:8001/${result.overlay_image}`,
                caseId: result.case_id,
                tumorDetected: result.tumor_detected,
                confidence: result.confidence,
                stage: result.stage,
                date: new Date().toLocaleString()
            };

            historyData.unshift(entry);
            localStorage.setItem("analysisHistory", JSON.stringify(historyData));
            renderHistory();
        };
        reader.readAsDataURL(file);

        // Display mask and overlay from backend with cache-busting
        const timestamp = Date.now();
        document.getElementById("maskImage").src = `http://127.0.0.1:8001/${result.mask_image}?t=${timestamp}`;
        document.getElementById("overlayImage").src = `http://127.0.0.1:8001/${result.overlay_image}?t=${timestamp}`;
        
        // Show results summary
        alert(`Analysis Complete!\nTumor Detected: ${result.tumor_detected ? 'Yes' : 'No'}\nConfidence: ${(result.confidence * 100).toFixed(1)}%\nStage: ${result.stage}`);

    } catch (error) {
        console.error('Error:', error);
        alert('Error analyzing image. Please make sure the backend is running on http://127.0.0.1:8001');
        
        // Fallback: display original image
        const reader = new FileReader();
        reader.onload = function (e) {
            const imageSrc = e.target.result;
            document.getElementById("inputImage").src = imageSrc;
            document.getElementById("maskImage").src = imageSrc;
            document.getElementById("overlayImage").src = imageSrc;
        };
        reader.readAsDataURL(file);
    }
}

function renderHistory() {
    const historyDiv = document.getElementById("history");
    historyDiv.innerHTML = "";

    historyData.forEach((item, index) => {
        const card = document.createElement("div");
        card.className = "history-card";

        card.innerHTML = `
            <img src="${item.image}">
            <p>${item.date}</p>
        `;

        card.onclick = () => loadFromHistory(index);
        historyDiv.appendChild(card);
    });
}

function loadFromHistory(index) {
    const item = historyData[index];
    document.getElementById("inputImage").src = item.image;
    document.getElementById("maskImage").src = item.maskImage || item.image;
    document.getElementById("overlayImage").src = item.overlayImage || item.image;
}

renderHistory();

// Batch processing functions
function enableBatchUpload() {
    isBatchMode = true;
    document.getElementById("batchFileInput").click();
}

function handleBatchFileSelection() {
    const batchFileInput = document.getElementById("batchFileInput");
    batchFiles = Array.from(batchFileInput.files);
    
    if (batchFiles.length === 0) return;
    
    // Show batch preview
    document.getElementById("batchPreview").style.display = "block";
    document.getElementById("batchButton").style.display = "inline-block";
    document.getElementById("fileCount").textContent = batchFiles.length;
    
    // Display file list
    const fileList = document.getElementById("fileList");
    fileList.innerHTML = "";
    
    batchFiles.forEach((file, index) => {
        const fileItem = document.createElement("div");
        fileItem.style.cssText = "padding: 5px; margin: 2px 0; background: #f9f9f9; border-radius: 4px; font-size: 12px;";
        fileItem.textContent = `${index + 1}. ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
        fileList.appendChild(fileItem);
    });
}

async function analyzeBatch() {
    if (batchFiles.length === 0) {
        alert("Please select files for batch processing.");
        return;
    }
    
    // Switch to batch results view
    document.getElementById("singleResults").style.display = "none";
    document.getElementById("batchResults").style.display = "block";
    document.getElementById("batchResultsContainer").innerHTML = "";
    
    const batchResults = [];
    let processedCount = 0;
    let tumorDetectedCount = 0;
    
    // Process each file
    for (let i = 0; i < batchFiles.length; i++) {
        const file = batchFiles[i];
        
        try {
            const result = await processSingleFile(file);
            batchResults.push(result);
            
            if (result.tumorDetected) {
                tumorDetectedCount++;
            }
            
            // Display result immediately
            displayBatchResult(result, i);
            processedCount++;
            
            // Update progress
            updateBatchProgress(processedCount, batchFiles.length, tumorDetectedCount);
            
        } catch (error) {
            console.error(`Error processing ${file.name}:`, error);
            batchResults.push({
                fileName: file.name,
                error: error.message,
                tumorDetected: false
            });
            processedCount++;
            updateBatchProgress(processedCount, batchFiles.length, tumorDetectedCount);
        }
    }
    
    saveBatchToHistory(batchResults);
    
    setTimeout(() => {
        alert(`Batch processing complete!\nProcessed: ${processedCount}/${batchFiles.length} files\nTumors detected: ${tumorDetectedCount}`);
    }, 500);
}

// Add event listener for batch file input
document.getElementById("batchFileInput").addEventListener("change", handleBatchFileSelection);
