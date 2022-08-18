const loaderContainer = document.querySelector('.loader-container');
const btn = document.querySelector('.news__btn');
const form = document.getElementById('form');

btn.addEventListener('click', (event) => {
    loaderContainer.classList.toggle('loader-container--hidden');
    form.submit()
});