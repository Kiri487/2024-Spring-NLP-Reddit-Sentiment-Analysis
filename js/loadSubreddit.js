document.querySelectorAll('.class-list-subreddit').forEach(element => {
    element.addEventListener('click', function() {  
      const filePath = this.getAttribute('data-file');
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
    });
});