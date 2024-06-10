document.addEventListener('DOMContentLoaded', function() {
  const projectIntros = document.querySelectorAll('.project-intro');
  const classListSubreddits = document.querySelectorAll('.class-list-subreddit');

  projectIntros.forEach(element => {
    element.addEventListener('click', function() {
      projectIntros.forEach(el => el.classList.remove('active-project-intro'));
      classListSubreddits.forEach(el => el.classList.remove('active-class-list-subreddit'));

      this.classList.add('active-project-intro');

      loadContent(this);
    });
  });

  classListSubreddits.forEach(element => {
    element.addEventListener('click', function() {
      projectIntros.forEach(el => el.classList.remove('active-project-intro'));
      classListSubreddits.forEach(el => el.classList.remove('active-class-list-subreddit'));

      this.classList.add('active-class-list-subreddit');

      loadContent(this);
    });
  });

  function loadContent(element) {
    const filePath = element.getAttribute('data-file');
    fetch(filePath)
      .then(response => {
        if (response.ok) return response.text();
        throw new Error('Failed to load the page');
      })
      .then(html => {
        document.getElementById('subreddit').innerHTML = html;
      })
      .catch(error => {
        console.error(error);
        document.getElementById('subreddit').innerHTML = '<p>Error loading content.</p>';
      });
  }

  const defaultOption = document.querySelector('.project-intro[data-file="./about.html"]');
  if (defaultOption) {
    defaultOption.click();
  }
});