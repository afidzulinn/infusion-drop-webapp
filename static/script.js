document.addEventListener('DOMContentLoaded', function() {
    const startBtn = document.getElementById('start-btn');
    const stopBtn = document.getElementById('stop-btn');
    const totalDropsSpan = document.getElementById('totalDrops');
    const timeBetweenDropsSpan = document.getElementById('timeBetweenDrops');

    // startBtn.addEventListener('click', function() {
    //     fetch('/video_feed', { method: 'POST' })
    //         .then(response => response.json())
    //         .then(data => console.log(data))
    //         .catch(error => console.error('Error:', error));

    //     stopBtn.style.display = 'block';
    //     startBtn.style.display = 'none';
    // });

    startBtn.addEventListener('click', function() {
        fetch('/video_feed')
            .then(response => response.json())
            .then(data => console.log(data))
            .catch(error => console.error('Error:', error));
            console.log();

        stopBtn.style.display = 'block';
        startBtn.style.display = 'block';

        window.location.reload();
    });

    stopBtn.addEventListener('click', function() {
        fetch('/stop_camera', { method: 'POST' })
            .then(response => response.json())
            .then(data => console.log(data))
            .catch(error => console.error('Error:', error));

        startBtn.style.display = 'block';
        stopBtn.style.display = 'block';
    });

    setInterval(() => {
        fetch('/drop_stats')
            .then(response => response.json())
            .then(data => {
                totalDropsSpan.textContent = data.total_drops;
                timeBetweenDropsSpan.textContent = data.drops_in_one_minute;
            })
            .catch(error => console.error('Error fetching stats:', error));
    }, 58000); // ms
});
