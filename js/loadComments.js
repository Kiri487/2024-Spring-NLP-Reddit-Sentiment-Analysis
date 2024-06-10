let commentUpdateInterval;

async function loadComments(jsonPath) {
  try {
    const response = await fetch(jsonPath);
    if (!response.ok) throw new Error('Failed to fetch comments');
    const comments = await response.json();
    let currentIndex = 0;

    clearInterval(commentUpdateInterval);

    function updateComments() {
      for (let i = 1; i <= 5; i++) {
        const commentElement = document.getElementById(`comment-${i}`);
        if (commentElement) {

          commentElement.classList.add('fade-out');
          setTimeout(() => {
            commentElement.innerText = comments[(currentIndex + i - 1) % comments.length].sentence;

            commentElement.classList.remove('fade-out');
            commentElement.classList.add('fade-in');

            setTimeout(() => {
              commentElement.classList.remove('fade-in');
            }, 500);
          }, 500);
        }
      }
      currentIndex = (currentIndex + 1) % comments.length;
    }

    updateComments();
    commentUpdateInterval = setInterval(updateComments, 3000);
  } catch (error) {
    console.error('Error loading comments:', error);
  }
}