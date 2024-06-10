document.addEventListener('DOMContentLoaded', function() {
  const projectIntros = document.querySelectorAll('.project-intro');
  const classListSubreddits = document.querySelectorAll('.class-list-subreddit');

  function toggleActiveClass(projectIntros, classListSubreddits, clickedElement, activeClass) {
    projectIntros.forEach(el => el.classList.remove('active-project-intro'));
    classListSubreddits.forEach(el => el.classList.remove('active-class-list-subreddit'));
    clickedElement.classList.add(activeClass);
  }

  function loadContent(element, loadCommentsFlag) {
    const filePath = element.getAttribute('data-file');
    fetch(filePath)
      .then(response => {
        if (response.ok) return response.text();
        throw new Error('Failed to load the page');
      })
      .then(html => {
        document.getElementById('subreddit').innerHTML = html;
        if (loadCommentsFlag) {
          const jsonPath = element.getAttribute('data-json');
          if (jsonPath && typeof loadComments === 'function') {
            loadComments(jsonPath);
          }
        }
      })
      .catch(error => {
        console.error(error);
        document.getElementById('subreddit').innerHTML = '<p>Error loading content.</p>';
      });
  }

  projectIntros.forEach(element => {
    element.addEventListener('click', function() {
      toggleActiveClass(projectIntros, classListSubreddits, this, 'active-project-intro');
      loadContent(this, false);
    });
  });

  classListSubreddits.forEach(element => {
    element.addEventListener('click', function() {
      toggleActiveClass(projectIntros, classListSubreddits, this, 'active-class-list-subreddit');
      loadContent(this, true);
    });
  });

  const defaultOption = document.querySelector('.project-intro[data-file="./about.html"]');
  if (defaultOption) {
    defaultOption.click();
  }
});