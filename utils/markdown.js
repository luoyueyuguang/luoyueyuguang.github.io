/**
 * 通用 Markdown 解析工具
 * 提供 Markdown 文件加载、解析和显示的通用功能
 * 供 pitfalls 和 learning 模块共享使用
 */

/**
 * 加载 Markdown 文件内容
 * @param {string} filePath - Markdown 文件路径
 * @returns {Promise<string>} Markdown 文件内容
 */
async function loadMarkdown(filePath) {
    try {
        const response = await fetch(filePath);
        if (!response.ok) {
            throw new Error(`文件加载失败: ${response.status} ${response.statusText}`);
        }
        return await response.text();
    } catch (error) {
        console.error('加载 Markdown 文件时出错:', error);
        throw new Error(`无法加载文件: ${filePath}`);
    }
}

/**
 * 解析 Markdown 内容为 HTML
 * @param {string} markdown - Markdown 字符串
 * @returns {string} 解析后的 HTML
 */
function parseMarkdown(markdown) {
    try {
        // 检查 marked.js 是否已加载
        if (typeof marked === 'undefined') {
            throw new Error('marked.js 库未加载，请确保已引入 marked.min.js');
        }
        
        // 使用 marked.js 解析 Markdown
        return marked.parse(markdown);
    } catch (error) {
        console.error('解析 Markdown 时出错:', error);
        throw new Error('Markdown 解析失败');
    }
}

/**
 * 在指定容器中显示 HTML 内容
 * @param {string} html - 要显示的 HTML 内容
 * @param {HTMLElement} container - 显示内容的容器元素
 */
function displayMarkdown(html, container) {
    if (!container) {
        throw new Error('容器元素未找到');
    }
    
    // 移除加载状态
    container.classList.remove('loading');
    
    // 设置 HTML 内容
    container.innerHTML = html;
    
    // 添加内容样式类
    container.classList.add('content');
}

/**
 * 在指定容器中显示错误信息
 * @param {string} errorMessage - 错误信息
 * @param {HTMLElement} container - 显示错误的容器元素
 */
function displayError(errorMessage, container) {
    if (!container) {
        console.error('错误容器未找到:', errorMessage);
        return;
    }
    
    // 移除加载状态
    container.classList.remove('loading');
    
    // 添加错误状态样式
    container.classList.add('error');
    
    // 显示错误信息
    container.innerHTML = `
        <div class="error-message">
            <h3>加载错误</h3>
            <p>${errorMessage}</p>
        </div>
    `;
}

/**
 * 通用的 Markdown 文件加载和显示函数
 * 整合了完整的加载、解析和显示流程
 * @param {string} filePath - Markdown 文件路径
 * @param {HTMLElement} container - 显示内容的容器元素
 * @returns {Promise<void>}
 */
async function loadAndDisplayMarkdown(filePath, container) {
    try {
        // 检查参数
        if (!filePath) {
            displayError('未指定要查看的文件', container);
            return;
        }
        
        if (!container) {
            throw new Error('容器元素未找到');
        }
        
        // 加载 Markdown 文件
        const markdown = await loadMarkdown(filePath);
        
        // 解析 Markdown 为 HTML
        const html = parseMarkdown(markdown);
        
        // 显示解析后的内容
        displayMarkdown(html, container);
        
    } catch (error) {
        console.error('加载和显示 Markdown 时出错:', error);
        displayError(error.message || error, container);
    }
}

// 将函数暴露到全局作用域，以便在其他脚本中使用
if (typeof window !== 'undefined') {
    window.loadMarkdown = loadMarkdown;
    window.parseMarkdown = parseMarkdown;
    window.displayMarkdown = displayMarkdown;
    window.displayError = displayError;
    window.loadAndDisplayMarkdown = loadAndDisplayMarkdown;
}