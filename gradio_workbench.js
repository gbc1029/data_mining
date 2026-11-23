const workspace = element.closest('#scholar-workspace');
const form = element.querySelector('#scholar-search');
const question = form.querySelector('#scholar-question');
const search = form.querySelector('#search-button');
const clear = form.querySelector('#clear-question');
const interests = new Map();

function syncQuery() {
    clear.hidden = !question.value;
}

function renderInterests() {
    const tags = workspace.querySelector('#interest-tags');
    const buttons = Array.from(interests.values(), (choice) => {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'scholar-interest-tag';
        button.dataset.removeInterest = choice.value;
        button.setAttribute('aria-label', `移除兴趣 ${choice.dataset.label}`);
        button.textContent = choice.dataset.label;
        button.append(clear.querySelector('.scholar-icon').cloneNode(true));
        return button;
    });
    tags.replaceChildren(...buttons);
}

function updateInterest(choice) {
    const notice = workspace.querySelector('#interest-notice');
    const limit = Number(workspace.querySelector('#interest-panel').dataset.maxInterests);
    if (choice.checked && !interests.has(choice.value) && interests.size >= limit) {
        choice.checked = false;
        notice.textContent = `最多选择 ${limit} 个兴趣领域。`;
        return;
    }
    if (choice.checked) interests.set(choice.value, choice);
    else interests.delete(choice.value);
    notice.textContent = '';
    renderInterests();
}

function syncMode() {
    const mode = workspace.querySelector('#reader-mode');
    for (const button of workspace.querySelectorAll('button[data-mode]')) {
        button.setAttribute('aria-pressed', String(button.dataset.mode === mode.value));
    }
}

question.addEventListener('input', syncQuery);
clear.addEventListener('click', () => {
    question.value = '';
    syncQuery();
    question.focus();
});

workspace.addEventListener('click', (event) => {
    const example = event.target.closest('button[data-example]');
    const mode = event.target.closest('button[data-mode]');
    const removal = event.target.closest('button[data-remove-interest]');
    if (example) {
        question.value = example.dataset.example;
        syncQuery();
        question.focus();
    } else if (mode) {
        workspace.querySelector('#reader-mode').value = mode.dataset.mode;
        syncMode();
    } else if (removal) {
        const checkbox = interests.get(removal.dataset.removeInterest);
        if (checkbox) {
            checkbox.checked = false;
            updateInterest(checkbox);
            workspace.querySelector('#interest-panel summary').focus();
        }
    }
});

workspace.addEventListener('change', (event) => {
    if (event.target.id === 'reader-mode') {
        syncMode();
    } else if (event.target.matches('input[name="interest"]')) {
        updateInterest(event.target);
    }
});

form.addEventListener('submit', (event) => {
    event.preventDefault();
    const text = question.value.trim();
    if (search.disabled) return;
    if (!text) {
        question.value = '';
        question.reportValidity();
        return;
    }
    trigger('submit', {
        question: text,
        mode: workspace.querySelector('#reader-mode').value,
        interests: Array.from(interests.keys(), Number),
    });
});

const searchText = search.querySelector('span:last-child');
const observer = new MutationObserver(() => {
    if (!workspace.isConnected) return observer.disconnect();
    const busy = !!workspace.querySelector('#scholar-result [data-search-pending]');
    if (search.disabled !== busy) {
        search.disabled = busy;
        searchText.textContent = busy ? '检索中…' : '搜索';
    }
});
observer.observe(workspace, { childList: true, subtree: true });
